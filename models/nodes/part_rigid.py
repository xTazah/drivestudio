"""
PartRigidNodes — part-based rigid pedestrian model for the thesis Method M.

Each pedestrian instance is decomposed into 10 body segments (per spec §3).
Each segment is a rigid Gaussian cloud whose pose is computed each frame by
forward kinematics on the SMPL 24-joint kintree, using the data SMPL pose
(no LBS skinning). Per-instance per-segment 6-DoF residuals correct
systematic SMPL-estimation bias; M-noref freezes these residuals at identity.
"""
import logging

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
