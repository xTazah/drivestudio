#!/usr/bin/env bash
# scripts/train_thesis_queue.sh
#
# Sequential trainer for the thesis ablation table — NO train/test split rerun.
# Queue: 2 scenes x 4 methods (OmniRe / B0 / M-noref / M) x 1 camera = 8 runs.
# Run from repo root (/mnt/d/Git/drivestudio) inside the `drivestudio` conda env.
#
# Resumes after crash: if checkpoint_final.pth already exists for a (scene, method) tuple,
# that run is skipped. Per-run failures (e.g. CUDA OOM) are logged and the queue moves on.
#
# Protocol (applied to all 8 runs):
#   - Full scene (start=0, end=-1)
#   - NO train/test split: test_image_stride=0  (all frames used for training)
#   - Floater fix: depth.w=0.05, Background cull_alpha_thresh=0.01, cull_scale_thresh=0.3
#   - preload_device=cpu (lets 8 GB VRAM fit)
#   - NO videos rendered: no test-set video, no full-set video, no novel-view video.
#     Full-image / dynamic-only / human-only / vehicle-only metrics ARE still computed
#     and saved (metrics{,_eval}/images_full_*.json) via the render_full_video=False
#     decoupling added to tools/eval.py.
#
# Usage:
#   bash scripts/train_thesis_queue.sh             # run full queue
#   bash scripts/train_thesis_queue.sh --dry-run   # print what would run, do nothing

set -u  # error on unset vars; do NOT use -e because we want to continue past failures

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- env setup ------------------------------------------------------------
# Assume `conda activate drivestudio` was done by the user already.
# Re-export defensively in case PYTHONPATH or LD_LIBRARY_PATH got stomped.
export PYTHONPATH="$REPO_ROOT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

# --- queue config ---------------------------------------------------------
# Each queue entry is: "scene_id|cams|method"
#   cams = 1 (1-cam only for this rerun); method ∈ {OmniRe, B0, Mnoref, M}.
# Order: per scene, train OmniRe -> B0 -> M-noref -> M; scene 327 then scene 552.
QUEUE=(
    "327|1|OmniRe"
    "327|1|B0"
    "327|1|Mnoref"
    "327|1|M"
    "552|1|OmniRe"
    "552|1|B0"
    "552|1|Mnoref"
    "552|1|M"
)

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

# --- output paths ---------------------------------------------------------
# New location/name for this no-split rerun so it does not collide with the
# earlier test-split batches.
RUN_BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_OUT="${HOME}/logs/thesis_nosplit_batch_${RUN_BATCH_TS}"
mkdir -p "$BATCH_OUT"
SUMMARY_CSV="$BATCH_OUT/summary.csv"
echo "run_idx,scene,cams,method,status,wall_seconds,full_psnr,full_ssim,full_lpips,dynamic_psnr,dynamic_ssim,human_psnr,human_ssim,vehicle_psnr,vehicle_ssim,log_path" > "$SUMMARY_CSV"

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "================================================================"
echo "Thesis-protocol training queue (NO train/test split)"
echo "Repo:      $REPO_ROOT"
echo "Git SHA:   $GIT_SHA"
echo "Batch:     $BATCH_OUT"
echo "Queue size: ${#QUEUE[@]} runs"
echo "Dry run:   $DRY_RUN"
echo "Disk free: $(df -h "$REPO_ROOT" | awk 'NR==2 {print $4}')"
echo "================================================================"

# --- per-run launcher ------------------------------------------------------
# Args: scene_id  cams  method  run_idx  total
run_one() {
    local scene="$1"
    local cams="$2"
    local method="$3"
    local idx="$4"
    local total="$5"

    # Resolve the dataset variant.
    # 1cam variants: 1cams_B0.yaml (B0), 1cams_M.yaml (M and M-noref), 1cams.yaml (OmniRe).
    local dataset_variant
    case "$method" in
        B0)            dataset_variant="waymo/${cams}cams_B0" ;;
        M|Mnoref)      dataset_variant="waymo/${cams}cams_M"  ;;
        OmniRe)        dataset_variant="waymo/${cams}cams"    ;;
        *)             echo "BUG: unknown method $method"; return 99 ;;
    esac

    # Resolve the trainer/model config.
    # B0 uses streetgs.yaml (only RigidNodes block). M / Mnoref use method_M.yaml. OmniRe uses omnire.yaml.
    local config_file
    case "$method" in
        B0)            config_file="configs/streetgs.yaml" ;;
        M|Mnoref)      config_file="configs/method_M.yaml" ;;
        OmniRe)        config_file="configs/omnire.yaml"   ;;
    esac

    # Run name & output paths. New project namespace for the no-split rerun.
    local pad_idx="$(printf '%02d' "$idx")"
    local run_name="scene${scene}_${cams}cam_${method}_nosplit"
    local project_name="thesis_nosplit_${cams}cam_${method}"
    local output_root="${HOME}/logs/thesis_nosplit_runs"
    local run_dir="$output_root/$project_name/$run_name"
    local run_log="$BATCH_OUT/${pad_idx}_${run_name}.log"
    local ckpt_final="$run_dir/checkpoint_final.pth"

    echo
    echo "----------------------------------------------------------------"
    echo "[$idx/$total] scene=$scene  cams=$cams  method=$method"
    echo "             config=$config_file  dataset=$dataset_variant"
    echo "             run_dir=$run_dir"
    echo "             log=$run_log"

    # Resume / skip logic.
    if [ -f "$ckpt_final" ]; then
        echo "[$idx/$total] SKIP — checkpoint_final.pth already exists."
        echo "$idx,$scene,$cams,$method,SKIPPED,0,,,,,,,,,," >> "$SUMMARY_CSV"
        return 0
    fi

    # Build the CLI override list.
    # --- shared overrides for all runs ---
    local cli_args=(
        "--config_file" "$config_file"
        "--output_root" "$output_root"
        "--project" "$project_name"
        "--run_name" "$run_name"
        "dataset=$dataset_variant"
        "data.scene_idx=$scene"
        "data.start_timestep=0"
        "data.end_timestep=-1"
        "data.preload_device=cpu"
        # NO train/test split: use every frame for training, hold out none.
        "data.pixel_source.test_image_stride=0"
        # floater fix
        "trainer.losses.depth.w=0.05"
        "model.Background.ctrl.cull_alpha_thresh=0.01"
        "model.Background.ctrl.cull_scale_thresh=0.3"
        # --- NO videos of any kind ---
        # no test-set video (and there is no test set anyway with stride=0)
        "render.render_test=False"
        # still compute full-set metrics (full / dynamic / human / vehicle)...
        "render.render_full=True"
        # ...but skip the heavy full-set video that OOM-kills the server.
        "render.render_full_video=False"
        # no novel-view rendering
        "render.render_novel.traj_types=[]"
    )

    # --- method-specific overrides ---
    if [ "$method" = "Mnoref" ]; then
        cli_args+=("model.PartRigidNodes.ctrl.refine_pose=False")
    fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY RUN — would execute:"
        echo "    python tools/train.py ${cli_args[*]}"
        echo "$idx,$scene,$cams,$method,DRY_RUN,0,,,,,,,,,," >> "$SUMMARY_CSV"
        return 0
    fi

    # Launch.
    local start_epoch
    start_epoch=$(date +%s)
    {
        echo "==== thesis-nosplit-batch run ${idx}/${total} ===="
        echo "Timestamp: $(date -Iseconds)"
        echo "Git SHA:   $GIT_SHA"
        echo "Command:   python tools/train.py ${cli_args[*]}"
        echo "========================================="
    } > "$run_log"

    if python tools/train.py "${cli_args[@]}" >>"$run_log" 2>&1; then
        local status="OK"
    else
        local rc=$?
        local status="FAILED_rc${rc}"
        echo "[$idx/$total] FAILED rc=$rc — see $run_log"
    fi
    local end_epoch
    end_epoch=$(date +%s)
    local wall=$(( end_epoch - start_epoch ))

    # Parse final eval metrics out of the log (best-effort). These lines come from
    # models/video_utils.py render_images() and cover every metric group asked for:
    #   Full Image PSNR/SSIM/LPIPS, Dynamic-Only PSNR/SSIM, Human-Only PSNR/SSIM,
    #   Vehicle-Only PSNR/SSIM.
    local full_psnr full_ssim full_lpips
    local dynamic_psnr dynamic_ssim human_psnr human_ssim vehicle_psnr vehicle_ssim
    full_psnr=$(grep -E "Full Image  PSNR:"  "$run_log" | tail -n1 | awk '{print $NF}')
    full_ssim=$(grep -E "Full Image  SSIM:"  "$run_log" | tail -n1 | awk '{print $NF}')
    full_lpips=$(grep -E "Full Image LPIPS:" "$run_log" | tail -n1 | awk '{print $NF}')
    dynamic_psnr=$(grep -E "Dynamic-Only PSNR:" "$run_log" | tail -n1 | awk '{print $NF}')
    dynamic_ssim=$(grep -E "Dynamic-Only SSIM:" "$run_log" | tail -n1 | awk '{print $NF}')
    human_psnr=$(grep -E "Human-Only PSNR:"   "$run_log" | tail -n1 | awk '{print $NF}')
    human_ssim=$(grep -E "Human-Only SSIM:"   "$run_log" | tail -n1 | awk '{print $NF}')
    vehicle_psnr=$(grep -E "Vehicle-Only PSNR:" "$run_log" | tail -n1 | awk '{print $NF}')
    vehicle_ssim=$(grep -E "Vehicle-Only SSIM:" "$run_log" | tail -n1 | awk '{print $NF}')
    full_psnr="${full_psnr:-NA}";       full_ssim="${full_ssim:-NA}";       full_lpips="${full_lpips:-NA}"
    dynamic_psnr="${dynamic_psnr:-NA}"; dynamic_ssim="${dynamic_ssim:-NA}"
    human_psnr="${human_psnr:-NA}";     human_ssim="${human_ssim:-NA}"
    vehicle_psnr="${vehicle_psnr:-NA}"; vehicle_ssim="${vehicle_ssim:-NA}"

    echo "$idx,$scene,$cams,$method,$status,$wall,$full_psnr,$full_ssim,$full_lpips,$dynamic_psnr,$dynamic_ssim,$human_psnr,$human_ssim,$vehicle_psnr,$vehicle_ssim,$run_log" >> "$SUMMARY_CSV"
    echo "[$idx/$total] $status  wall=${wall}s  Full=$full_psnr/$full_ssim/$full_lpips  Dyn=$dynamic_psnr/$dynamic_ssim  Human=$human_psnr/$human_ssim  Veh=$vehicle_psnr/$vehicle_ssim"
}

# --- main loop ------------------------------------------------------------
TOTAL=${#QUEUE[@]}
IDX=0
for entry in "${QUEUE[@]}"; do
    [ -z "$entry" ] && continue
    IDX=$(( IDX + 1 ))
    IFS='|' read -r scene cams method <<< "$entry"
    run_one "$scene" "$cams" "$method" "$IDX" "$TOTAL"
done

echo
echo "================================================================"
echo "Batch complete. Summary:"
column -s, -t < "$SUMMARY_CSV"
echo "Full summary: $SUMMARY_CSV"
echo "================================================================"
