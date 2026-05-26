# SETUP — DriveStudio fork (xTazah)

End-to-end workflow for running this fork on **Waymo Open Dataset** scenes
using the **Street Gaussians** config (`configs/streetgs.yaml`).

Scope: this doc assumes the conda environments already exist. If they don't,
see [Conda envs](#conda-envs) below for which versions to recreate.

This doc captures the things that **differ from upstream
[docs/Waymo.md](docs/Waymo.md)**, including:

- Two patches we apply at runtime (see [Known patches](#known-patches))
- Which mmcv version actually works (the doc's pin is broken)
- Override flags to make 1-camera training fit on an 8 GB GPU
- A streaming patch in `models/video_utils.py` that bounds eval CPU memory

---

## Conda envs

| Env name      | Python | Purpose                              | Activate before                          |
| ------------- | ------ | ------------------------------------ | ---------------------------------------- |
| `drivestudio` | 3.9    | Preprocess, train, eval the model    | Steps 2, 4, 5 below                      |
| `segformer`   | 3.8    | Generate sky masks (and fine dynamic masks if needed) | Step 3 below |

The two envs are isolated on purpose — `segformer` needs torch 1.8 + mmcv-full
1.3.x and would conflict with `drivestudio`'s torch 2.0 stack.

### Shared shell prereqs (every WSL session)

```bash
# Re-export for every new shell. Add to ~/.bashrc if it gets old.
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

WSL's CUDA driver (`libcuda.so.1`) lives at `/usr/lib/wsl/lib/` and isn't on
the default linker path. Without this export, GPU rendering crashes with
`Could not load library libcudnn_cnn_infer.so.8`.

Both envs' `activate.d/wsl_cuda.sh` already do this on `conda activate`, but
external scripts may override `LD_LIBRARY_PATH`, so re-export defensively.

---

## Step 1 — Pick a scene

The canonical scene list is `data/waymo_train_list.txt`. The scene index is
**line number minus one**. Example: line 24 → `scene_idx=23` → segment
`seg104554`.

For reproduction comparisons against the OmniRe paper, use scenes from
`data/waymo_example_scenes.txt`:

```
scene_id  seg_name   start_timestep  end_timestep
23        seg104554  0               150
114       seg125050  0               150
327       seg169514  0               150
621       seg584622  0               140
703       seg776165  0               170
172       seg138251  30              180
552       seg448767  0               140
788       seg965324  130             -1
```

The rest of this doc uses `SCENE_IDX=23` and `START=0`, `END=150`. Substitute
your scene throughout.

---

## Step 2 — Download + preprocess (env: `drivestudio`)

```bash
conda activate drivestudio
cd ~/drivestudio  # or wherever your clone lives — adjust paths below
export PYTHONPATH=$(pwd)
```

### 2a. Download the raw tfrecord

```bash
mkdir -p ./data/waymo/raw ./data/waymo/processed
python datasets/waymo/waymo_download.py \
    --target_dir ./data/waymo/raw \
    --scene_ids 23
```

~1 GB per scene at home internet speed.

If gcloud auth has expired:
```bash
gcloud auth login
gcloud auth application-default login
```

Quick sanity check that your account is approved:
```bash
gsutil ls gs://waymo_open_dataset_scene_flow/ | head -5
```

### 2b. Extract per-frame data

```bash
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --scene_ids 23 \
    --workers 4 \
    --process_keys images lidar calib pose dynamic_masks objects
```

**Why this set of `--process_keys`:** matches what `configs/streetgs.yaml`
loads. Notably we skip `humanpose` — streetgs doesn't use SMPL nodes (those
are an OmniRe feature for pedestrians). If you later train OmniRe, add
`humanpose` and do the SMPL preprocessing in [docs/HumanPose.md](docs/HumanPose.md).

**Why `--workers 4`:** more than that uses too much RAM on a 32 GB machine
when multiple tfrecords are loaded simultaneously.

Output appears under `data/waymo/processed/training/023/` — verify these
directories exist:
```bash
ls data/waymo/processed/training/023/
# Expected: dynamic_masks  ego_pose  extrinsics  frame_info.json  images
#           instances  intrinsics  lidar  sky_masks
```

**Note:** `sky_masks/` exists but is empty after this step. The main
preprocessing script creates the directory but doesn't populate it; that's
Step 3.

---

## Step 3 — Generate sky masks (env: `segformer`)

This step uses the **separate** `segformer` conda env. Don't try to install
SegFormer into the `drivestudio` env — they have incompatible torch/mmcv
versions.

### 3a. One-time: create and populate the `segformer` env

```bash
# 1. Clone SegFormer
git clone https://github.com/NVlabs/SegFormer ~/SegFormer
#   (on WSL: /mnt/d/Git/SegFormer — adjust segformer_path in 3b accordingly)

# 2. Create env
conda create -n segformer python=3.8 -y
conda activate segformer

# 3. PyTorch 1.8.1 + cu111
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 4. Other deps
pip install timm==0.3.2 pylint debugpy opencv-python-headless \
    attrs ipython tqdm imageio scikit-image omegaconf gdown

# 5. mmcv-full prebuilt wheel — NOT 1.2.7 from source (that fails; see gotchas)
pip install mmcv-full==1.3.18 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# 6. Install SegFormer
cd ~/SegFormer && pip install .

# 7. Patch the mmcv version cap — MUST target the site-packages copy, not
#    the source clone (pip install . copies mmseg there; patching the clone
#    has no effect on what Python imports)
sed -i "s/MMCV_MAX = '1.3.0'/MMCV_MAX = '1.4.0'/" \
    "$CONDA_PREFIX/lib/python3.8/site-packages/mmseg/__init__.py"

# 8. Download checkpoint (~970 MB)
mkdir -p ~/SegFormer/pretrained && cd ~/SegFormer/pretrained
gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj
# The file lands as segformer.b5.1024x1024.city.160k.pth
```

Verify the setup:
```bash
python -c "from mmseg.apis import inference_segmentor, init_segmentor; print('OK')"
# Expected: OK
```

### 3b. Run sky mask extraction

```bash
conda activate segformer
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd ~/drivestudio  # back to drivestudio repo

segformer_path=~/SegFormer  # adjust if clone is elsewhere

python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/training \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --scene_ids 23
```

**Runtime:** ~15 min per scene on RTX 2070 SUPER. Per-frame inference at
1920×1280 is 1–2 s; for 151 frames × 5 cameras that's ~25 min total. SegFormer-B5
uses ~2–4 GB VRAM at inference, fits comfortably on 8 GB cards.

**Why not `--process_dynamic_mask`:** for the streetgs baseline we use the
coarse dynamic masks Waymo provides (already populated in Step 2b). Add
`--process_dynamic_mask` if you want SegFormer's fine masks too, but it adds
~10 min and isn't required for streetgs.yaml.

Verify masks were written:
```bash
ls data/waymo/processed/training/023/sky_masks/ | wc -l
# Expected: num_frames_in_tfrecord * 5 cameras. The preprocessing extracts
# every frame in the tfrecord; training uses a sub-window via
# data.start/end_timestep. Compare to image count:
ls data/waymo/processed/training/023/images/ | wc -l
# The two should match. For scene 23 it's 995 (199 frames * 5 cams).
```

---

## Step 4 — Train (env: `drivestudio`)

```bash
conda activate drivestudio
cd ~/drivestudio
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### 4a. Smoke test (300 iters, ~5 min)

Use this to confirm everything still works after a code change or env update.

```bash
python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root logs/smoke_test \
    --project streetgs_smoke \
    --run_name scene23_smoke \
    dataset=waymo/1cams \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=50 \
    data.preload_device=cpu \
    model.Background.init.from_lidar.num_samples=200_000 \
    model.Background.init.near_randoms=20_000 \
    model.Background.init.far_randoms=20_000 \
    trainer.optim.num_iters=300 \
    logging.print_freq=50
```

**Why each override:**

- `dataset=waymo/1cams` — 1 camera instead of 3/5. On 8 GB VRAM, more cameras
  push the rasterizer past dedicated memory and start paging to host RAM
  (catastrophically slow).
- `data.start/end_timestep=0/50` — short clip. 50 frames is enough to confirm
  data loading + gradient flow without long preload time.
- `data.preload_device=cpu` — keep image cache in CPU RAM, not VRAM. Saves
  ~3–4 GB VRAM. Each step copies the active frame's data to GPU on demand.
  Slightly slower per step but fits on 8 GB.
- `model.Background.init.*` — reduces initial Background gaussian count from
  800k+200k+100k+100k ≈ 1M to ~240k. Densification will grow the count during
  training; starting smaller leaves headroom.
- `trainer.optim.num_iters=300` — smoke test only.
- `logging.print_freq=50` — see loss progression in real time instead of
  waiting for one print at the end.

Expected at iter 300: `train_metrics/psnr ≈ 26`. Eval should yield
`Full Image PSNR ≈ 26.3, SSIM ≈ 0.88, LPIPS ≈ 0.24`.

### 4b. Full training (30k iters, several hours)

For the real reproduction baseline, omit the iter cap and don't reduce
gaussian counts:

```bash
python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root logs/streetgs_baseline \
    --project streetgs_baseline \
    --run_name scene23_full \
    dataset=waymo/1cams \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=150 \
    data.preload_device=cpu
```

**What gets trained by default:** only vehicles. Out of the box, the Waymo
loader routes `Vehicle → RigidNodes`, `Pedestrian → SMPLNodes`,
`Cyclist → DeformableNodes`. `streetgs.yaml` only enables RigidNodes (no
SMPL or Deformable nodes), so pedestrians and cyclists are silently dropped
during scene-graph initialization. If you want a true "Street Gaussians
baseline" where *all* dynamic objects are treated as rigid (B0 in the
thesis ablation table), use the `waymo/1cams_B0` dataset variant:

```bash
python tools/train.py \
    --config_file configs/streetgs.yaml \
    ... (other args as above) ... \
    dataset=waymo/1cams_B0
```

`1cams_B0.yaml` is identical to `1cams.yaml` except it sets
`data.pixel_source.object_class_node_mapping` to route all dynamic classes
(Vehicle, Pedestrian, Cyclist) to RigidNodes.

After training, look for the log line near the start:

```
INFO root - Object class -> node-type mapping: {'Vehicle': 'RigidNodes', 'Pedestrian': 'RigidNodes', 'Cyclist': 'RigidNodes'}
```

That confirms the override took effect. Without it (e.g. running the
default `dataset=waymo/1cams`), your eval videos will show no pedestrians
and no cyclists in the dynamic layer — they're missing, not ghosted.

**Why a config variant instead of a CLI flag:** OmegaConf's
`from_cli` parser (used by `tools/train.py`) handles nested-key creation
inconsistently. Both `+key.subkey=val` and `++key.subkey=val` either error
or land the key in the wrong place. Using a YAML config variant sidesteps
the issue entirely. If you need ad-hoc per-run mappings beyond what
`1cams_B0.yaml` provides, copy that file and edit it.

For slow-walker peds, you may also want to relax the trajectory filter
(default `traj_length_thres: 1.0` excludes peds walking under 1 m within
the training window):

```bash
... dataset=waymo/1cams_B0 \
    model.RigidNodes.init.only_moving=False
```

On RTX 2070 SUPER: expect ~3–5 hours wall-time for 30k iters at 1 camera.
The ETA printed early in training underestimates — it doesn't account for
per-iter time growing as densification adds gaussians (from ~0.45 s/iter at
step 1000 to ~0.7–0.9 s/iter near step 15000, then stabilizing). VRAM use
also climbs from ~4 GB initially to ~6.5–7 GB by mid-training. GPU
utilization will sit around 60% because ~40% of each iteration is CPU-side
densification bookkeeping, not GPU compute — this is normal and not fixable
without rewriting the densification code.

**Known constraint:** novel-view rendering at end-of-training uses full
resolution and may exceed 8 GB VRAM (spills to shared GPU memory and slows
down significantly). If this is a problem, append
`render.render_novel.traj_types=[]` to skip novel views, or wait for the
followup patch that adds a `render.render_novel.downscale` knob.

---

## Step 5 — Evaluate a checkpoint (env: `drivestudio`)

```bash
conda activate drivestudio
cd ~/drivestudio
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

python tools/eval.py --resume_from logs/streetgs_baseline/streetgs_baseline/scene23_full/checkpoint_final.pth
```

Produces metrics JSONs + mp4s under `logs/.../videos_eval/`.

---

## Known patches

This fork has small patches over upstream. They are committed; you don't need
to re-apply them. Documented here so future code archaeologists understand
the deltas:

1. **`datasets/base/pixel_source.py`** — initialize `self.sky_masks = None`
   unconditionally so `to(device)` doesn't `AttributeError` when
   `load_sky_mask=False`.

2. **`models/trainers/base.py:compute_losses`** — graceful fallback when
   `sky_masks` key is absent from `image_infos`. Without this fix,
   `load_sky_mask=False` crashes at the first loss compute.

3. **`models/video_utils.py:render()`** — replaced in-memory frame-list
   accumulation with `RenderResults`/`_LazyFrameList` that streams per-frame
   tensors to a temp directory and serves them lazily. Eval CPU memory drops
   from ~25 GB to ~10 GB on a 51-frame 1-cam run. The
   `DRIVESTUDIO_RENDER_TMP` env var overrides the temp location if `/tmp` is
   too small.

4. **`tools/eval.py`** — wraps each `render_images` + `save_videos` pair in
   try/finally so the streaming patch's temp dir is always cleaned up,
   including on encoding failure.

5. **`datasets/waymo/waymo_sourceloader.py`** — class → node-type mapping
   is overridable via `data.pixel_source.object_class_node_mapping` in the
   config. Lets you route pedestrians and cyclists to RigidNodes (B0
   baseline) without editing source.

---

## Common gotchas

**`Could not load library libcudnn_cnn_infer.so.8`** → forgot the
`LD_LIBRARY_PATH=/usr/lib/wsl/lib:...` export. Add it before any GPU call.

**`AssertionError: MMCV==1.3.x is used but incompatible`** during SegFormer
mask extraction → the version cap in
`$CONDA_PREFIX/lib/python3.8/site-packages/mmseg/__init__.py` rejects newer
mmcv. We patched `MMCV_MAX = '1.4.0'` already; if you set up a fresh segformer
env, redo:

```bash
sed -i "s/MMCV_MAX = '1.3.0'/MMCV_MAX = '1.4.0'/" \
    /home/$USER/miniconda3/envs/segformer/lib/python3.8/site-packages/mmseg/__init__.py
```

**`mmcv-full` build fails with `Thrust requires at least C++17`** → don't
build from source. Use the prebuilt wheel:
```bash
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

**OOM during eval video rendering** → either WSL memory is too low (set
`memory=26GB` in `%UserProfile%\.wslconfig` on the Windows side and
`wsl --shutdown`) or you're hitting the production-scale problem the
streaming patch already addresses. If you still OOM, set
`DRIVESTUDIO_RENDER_TMP=/path/with/lots/of/space` to move the streaming
temp dir off `/tmp`.

**Novel-view rendering uses shared GPU memory and is slow** → the
`render_novel_views` function in `models/video_utils.py` ignores
`downscale_when_loading` and renders at full resolution. On 8 GB cards
this spills to shared memory. Workaround: append
`render.render_novel.traj_types=[]` to skip novel views.

---

## Where things live

| Path                                            | What it is                              |
| ----------------------------------------------- | --------------------------------------- |
| `~/drivestudio` (or your clone)                 | This fork                               |
| `~/drivestudio/data/waymo/raw/*.tfrecord`       | Downloaded Waymo segments               |
| `~/drivestudio/data/waymo/processed/training/`  | Preprocessed scenes                     |
| `/mnt/d/Git/SegFormer/`                         | SegFormer repo (separate)               |
| `/mnt/d/Git/SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth` | Sky-mask checkpoint (~970 MB) |
| `~/miniconda3/envs/drivestudio/`                | Main training env                       |
| `~/miniconda3/envs/segformer/`                  | Mask-extraction env                     |
| `/tmp/drivestudio_render_*` (transient)         | Streaming eval frames (auto-cleaned)    |
