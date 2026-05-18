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
