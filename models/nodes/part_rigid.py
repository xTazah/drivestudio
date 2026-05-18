"""
PartRigidNodes — part-based rigid pedestrian model for the thesis Method M.

Each pedestrian instance is decomposed into 10 body segments (per spec §3).
Each segment is a rigid Gaussian cloud whose pose is computed each frame by
forward kinematics on the SMPL 24-joint kintree, using the data SMPL pose
(no LBS skinning). Per-instance per-segment 6-DoF residuals correct
systematic SMPL-estimation bias; M-noref freezes these residuals at identity.
"""
import logging
from typing import Dict, List

import torch
from torch.nn import Parameter

from models.gaussians.basics import matrix_to_quaternion, quat_mult
from models.human_body import quaternion_to_matrix
from models.nodes.smpl import SMPLNodes

logger = logging.getLogger()

# ---------------------------------------------------------------------------
# Body segmentation topology (spec §3, N=10)
# ---------------------------------------------------------------------------
# For each of the 10 segments: (segment_name, anchor_joint_idx, [joints_folded_in])
SEGMENT_TABLE = [
    ("pelvis_spine",      0,  [0, 3, 6, 9]),    # pelvis + 3 spine joints
    ("head_neck",        12,  [12, 15]),         # neck + head
    ("L_upper_arm",      16,  [13, 16]),         # L collar + L shoulder
    ("R_upper_arm",      17,  [14, 17]),         # R collar + R shoulder
    ("L_forearm_hand",   18,  [18, 20, 22]),     # L elbow + L wrist + L hand
    ("R_forearm_hand",   19,  [19, 21, 23]),     # R elbow + R wrist + R hand
    ("L_upper_leg",       1,  [1]),              # L hip
    ("R_upper_leg",       2,  [2]),              # R hip
    ("L_lower_leg_foot",  4,  [4, 7, 10]),       # L knee + L ankle + L foot
    ("R_lower_leg_foot",  5,  [5, 8, 11]),       # R knee + R ankle + R foot
]

NUM_SEGMENTS = len(SEGMENT_TABLE)
assert NUM_SEGMENTS == 10

# Anchor joint index per segment, shape (10,)
SEGMENT_ANCHORS = torch.tensor([row[1] for row in SEGMENT_TABLE], dtype=torch.long)

# joint_to_segment[j] = segment id that owns joint j. Shape (24,).
_joint_to_segment = [-1] * 24
for seg_id, (_, _, joints) in enumerate(SEGMENT_TABLE):
    for j in joints:
        assert _joint_to_segment[j] == -1, f"joint {j} assigned twice"
        _joint_to_segment[j] = seg_id
assert all(s >= 0 for s in _joint_to_segment), f"unassigned joints: {_joint_to_segment}"
JOINT_TO_SEGMENT = torch.tensor(_joint_to_segment, dtype=torch.long)


def _forward_kinematics(
    rot_mats: torch.Tensor,
    joints_canonical: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-joint world-space SE(3) transforms along the SMPL kintree.

    Args:
        rot_mats: (B, 24, 3, 3) per-joint local rotation matrices.
        joints_canonical: (B, 24, 3) joint positions in T-pose.
        parents: (24,) long, parents[j] is parent joint index of joint j; parents[0] = -1.

    Returns:
        T_world: (B, 24, 4, 4). T_world[:, j] maps a point in joint j's local
                 (T-pose-centered) frame to canonical world space.
    """
    B, J = rot_mats.shape[0], rot_mats.shape[1]
    rel_joints = joints_canonical.clone()
    rel_joints[:, 1:] = rel_joints[:, 1:] - joints_canonical[:, parents[1:]]

    eye = torch.eye(4, dtype=rot_mats.dtype, device=rot_mats.device)
    local = eye.repeat(B, J, 1, 1)
    local[:, :, :3, :3] = rot_mats
    local[:, :, :3,  3] = rel_joints

    T_world = [local[:, 0]]
    for j in range(1, J):
        T_world.append(torch.matmul(T_world[parents[j]], local[:, j]))
    return torch.stack(T_world, dim=1)


def _apply_residuals(
    quats_24: torch.Tensor,
    seg_quat_residuals: torch.Tensor,
) -> torch.Tensor:
    """
    Compose per-segment residual quaternions onto the per-joint data quaternions
    at each segment's anchor joint only. Non-anchor joints pass through unchanged;
    the parent chain in FK propagates the residual to descendants in the same segment.

    Args:
        quats_24: (B, 24, 4) per-instance per-joint data quaternions.
        seg_quat_residuals: (B, 10, 4) residuals applied at anchors.

    Returns:
        (B, 24, 4) corrected quaternions.
    """
    out = quats_24.clone()
    eps = 1e-8
    for seg_id in range(NUM_SEGMENTS):
        anchor = SEGMENT_ANCHORS[seg_id].item()
        r = seg_quat_residuals[:, seg_id]
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(eps)
        d = quats_24[:, anchor]
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(eps)
        out[:, anchor] = quat_mult(r, d)
    return out


class PartRigidNodes(SMPLNodes):
    """Part-based rigid pedestrian Gaussians driven by SMPL FK."""

    def __init__(self, **kwargs):
        # Force off voxel deformer and ball gaussians: M is hard-rigid per segment.
        super().__init__(**kwargs)
        if getattr(self, "use_voxel_deformer", False):
            logger.warning("PartRigidNodes: use_voxel_deformer ignored (forced off)")
            self.use_voxel_deformer = False

        # Will be set in create_from_pcd / load_state_dict.
        self.point_segment_ids = None
        self.seg_quat_residuals = None
        self.seg_trans_residuals = None

    @property
    def refine_pose(self) -> bool:
        # Default True (full method M). Set to False in config for M-noref.
        return bool(self.ctrl_cfg.get("refine_pose", True))

    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        """
        Initialize segment topology, canonical-frame means, and residuals.

        Calls SMPLNodes.create_from_pcd for: SMPLTemplate construction, per-vertex
        initial means/scales/quats/opacities, parameter setup, instances_fv /
        instances_quats / instances_trans / smpl_qauts.

        On top of that we add:
          - point_segment_ids: (N_total_gaussians,) long
          - rewrite self._means.data so each vertex lives in its anchor joint's
            canonical (T-pose) frame
          - seg_quat_residuals, seg_trans_residuals parameters
        """
        super().create_from_pcd(instance_pts_dict)

        # 1. Per-vertex segment ids via SMPL skinning-weight argmax -> JOINT_TO_SEGMENT.
        # template.W shape: (num_instances, smpl_points_num, 24).
        W = self.template.W
        per_vertex_joint = W.argmax(dim=-1)                # (B, 6890)
        joint_to_seg_dev = JOINT_TO_SEGMENT.to(W.device)
        per_vertex_seg = joint_to_seg_dev[per_vertex_joint]  # (B, 6890)
        # _means is laid out (B*6890, 3); point_segment_ids parallel to that.
        self.point_segment_ids = per_vertex_seg.reshape(-1).contiguous()  # (B*6890,)

        # 2. Express each vertex in anchor-joint canonical frame.
        # template.J_canonical: (B, 24, 3) joint positions in T-pose.
        anchors_dev = SEGMENT_ANCHORS.to(W.device)
        anchor_joint_per_vertex = anchors_dev[per_vertex_seg]    # (B, 6890)
        J_can = self.template.J_canonical                        # (B, 24, 3)
        anchor_pos = torch.gather(
            J_can,
            dim=1,
            index=anchor_joint_per_vertex.unsqueeze(-1).expand(-1, -1, 3),
        )                                                        # (B, 6890, 3)
        anchor_pos_flat = anchor_pos.reshape(-1, 3)              # (B*6890, 3)
        with torch.no_grad():
            self._means.data = self._means.data - anchor_pos_flat

        # 3. Residual parameters: (B, 10, 4) quats init to identity, (B, 10, 3) trans init to zero.
        B = self.num_instances
        identity_q = torch.zeros(B, NUM_SEGMENTS, 4, device=self.device)
        identity_q[..., 0] = 1.0  # [1, 0, 0, 0]
        zero_t = torch.zeros(B, NUM_SEGMENTS, 3, device=self.device)
        self.seg_quat_residuals = Parameter(identity_q, requires_grad=self.refine_pose)
        self.seg_trans_residuals = Parameter(zero_t, requires_grad=self.refine_pose)

        logger.info(
            f"PartRigidNodes: {B} instances x {NUM_SEGMENTS} segments. "
            f"refine_pose={self.refine_pose}. "
            f"Per-segment vertex counts (instance 0): "
            f"{[int((per_vertex_seg[0] == s).sum()) for s in range(NUM_SEGMENTS)]}"
        )

    def transform_means_and_quats(
        self, means: torch.Tensor, quats: torch.Tensor,
    ):
        """
        Compute world-space means and orientations for all Gaussians at cur_frame
        via per-segment FK on the SMPL kintree.
        """
        assert means.shape[0] == self.point_ids.shape[0], "shape mismatch on means"
        cur = self.cur_frame
        instance_mask = self.instances_fv[cur]                # (B,)

        # Gather per-instance per-joint data quats: root + 23 body joints.
        root_q = self.instances_quats[cur]                     # (B, 1, 4)
        body_q = self.smpl_qauts[cur]                          # (B, 23, 4)
        full_q = torch.cat([root_q, body_q], dim=1)            # (B, 24, 4)
        full_q = self.quat_act(full_q)

        # Apply per-segment residuals (no-op if refine_pose=False — params frozen at identity).
        full_q = _apply_residuals(full_q, self.seg_quat_residuals)

        # FK along kintree.
        parents = self.template._template_layer.parents        # (24,)
        rot_mats = quaternion_to_matrix(full_q)                # (B, 24, 3, 3)
        J_can = self.template.J_canonical                      # (B, 24, 3)
        T_joint = _forward_kinematics(rot_mats, J_can, parents)  # (B, 24, 4, 4)

        # Per-instance world translation.
        trans_cur = self.instances_trans[cur]                  # (B, 3)

        # Per-segment world transforms: pick anchor-joint transform for each segment.
        anchors = SEGMENT_ANCHORS.to(T_joint.device)
        T_seg = T_joint[:, anchors].clone()                    # (B, 10, 4, 4)
        # Apply per-segment translation residual to anchor world position.
        T_seg[..., :3, 3] = T_seg[..., :3, 3] + self.seg_trans_residuals

        # Per-Gaussian transform lookup.
        ins_id = self.point_ids[..., 0]                        # (N,)
        seg_id = self.point_segment_ids                        # (N,)
        T_per_pt = T_seg[ins_id, seg_id]                       # (N, 4, 4)
        R_per_pt = T_per_pt[..., :3, :3]                       # (N, 3, 3)
        t_per_pt = T_per_pt[..., :3, 3]                        # (N, 3)
        t_root_per_pt = trans_cur[ins_id]                      # (N, 3)

        world_means = torch.bmm(R_per_pt, means.unsqueeze(-1)).squeeze(-1) + t_per_pt + t_root_per_pt

        R_quats = matrix_to_quaternion(R_per_pt)
        world_quats = quat_mult(self.quat_act(R_quats), self.quat_act(quats))

        # Mask out invisible instances.
        valid_per_pt = instance_mask[ins_id].unsqueeze(-1)
        world_means = world_means * valid_per_pt
        identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=world_quats.device).expand_as(world_quats)
        world_quats = torch.where(valid_per_pt.expand(-1, 4), world_quats, identity_q)

        return world_means, world_quats

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        # Delegate; supply a dummy identity-quat batch and discard the returned quats.
        dummy_q = torch.zeros(means.shape[0], 4, device=means.device)
        dummy_q[..., 0] = 1.0
        world_means, _ = self.transform_means_and_quats(means, dummy_q)
        return world_means

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        param_groups[self.class_prefix + "ins_rotation"]    = [self.instances_quats]
        param_groups[self.class_prefix + "ins_translation"] = [self.instances_trans]
        param_groups[self.class_prefix + "smpl_rotation"]   = [self.smpl_qauts]
        if self.refine_pose:
            param_groups[self.class_prefix + "seg_quat_residuals"]  = [self.seg_quat_residuals]
            param_groups[self.class_prefix + "seg_trans_residuals"] = [self.seg_trans_residuals]
        return param_groups

    def state_dict(self) -> Dict:
        state_dict = super().state_dict()
        state_dict.update({
            "point_segment_ids":   self.point_segment_ids,
            "seg_quat_residuals":  self.seg_quat_residuals.data,
            "seg_trans_residuals": self.seg_trans_residuals.data,
        })
        return state_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.point_segment_ids = state_dict.pop("point_segment_ids")
        seg_q = state_dict.pop("seg_quat_residuals")
        seg_t = state_dict.pop("seg_trans_residuals")
        self.seg_quat_residuals  = Parameter(seg_q, requires_grad=self.refine_pose)
        self.seg_trans_residuals = Parameter(seg_t, requires_grad=self.refine_pose)
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg

    # ------------------------------------------------------------------
    # Densification hooks: keep point_segment_ids in lockstep with point_ids.
    # ------------------------------------------------------------------
    def split_gaussians(self, split_mask: torch.Tensor, samps: int = 2):
        out = super().split_gaussians(split_mask, samps)
        new_segments = self.point_segment_ids[split_mask].repeat(samps)
        self.point_segment_ids = torch.cat([self.point_segment_ids, new_segments], dim=0)
        return out

    def dup_gaussians(self, dup_mask: torch.Tensor):
        out = super().dup_gaussians(dup_mask)
        new_segments = self.point_segment_ids[dup_mask]
        self.point_segment_ids = torch.cat([self.point_segment_ids, new_segments], dim=0)
        return out

    def cull_gaussians(self):
        """Re-implements RigidNodes.cull_gaussians to also mask point_segment_ids.

        Keep this in sync with RigidNodes.cull_gaussians (models/nodes/rigid.py).
        """
        n_bef = self.num_points
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            culls = culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values
                > self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means         = Parameter(self._means[~culls].detach())
        self._scales        = Parameter(self._scales[~culls].detach())
        self._quats         = Parameter(self._quats[~culls].detach())
        self._features_dc   = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities     = Parameter(self._opacities[~culls].detach())
        self.point_ids          = self.point_ids[~culls]
        self.point_segment_ids  = self.point_segment_ids[~culls]
        logger.info(f"     Cull: {n_bef - self.num_points}")
        return culls

    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        # Skip SMPLNodes' knn/voxel/joint_smooth reg losses (don't apply here).
        # Go straight to RigidNodes.compute_reg_loss for sharp_shape_reg and
        # the per-frame trans temporal smoothness.
        from models.nodes.rigid import RigidNodes
        loss_dict = RigidNodes.compute_reg_loss(self)

        residual_reg = self.reg_cfg.get("residual_reg", None)
        if residual_reg is not None and self.refine_pose:
            w_rot = residual_reg.get("w_rot", 0.0)
            w_trans = residual_reg.get("w_trans", 0.0)
            if w_rot > 0:
                # Penalize departure from identity quaternion [1,0,0,0].
                # 1 - q[0]^2 is proportional to ||R - I||^2 for unit quats.
                q = self.seg_quat_residuals
                q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                loss_dict["seg_residual_rot"] = (1.0 - q[..., 0] ** 2).mean() * w_rot
            if w_trans > 0:
                loss_dict["seg_residual_trans"] = self.seg_trans_residuals.pow(2).mean() * w_trans
        return loss_dict
