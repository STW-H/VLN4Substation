"""Region-goal, pose-aware A* planner implementation package."""

from substation_vln.planning.improved_astar.camera_model import CameraConfig
from substation_vln.planning.improved_astar.goal_pose_region import (
    build_pose_free_masks,
    generate_goal_pose_candidates,
    pack_pose_free_masks,
    quantize_heading,
    unpack_pose_free_masks,
)
from substation_vln.planning.improved_astar.pose_region_astar import (
    HierarchicalPoseAStarResult,
    PoseAStarConfig,
    PoseAStarResult,
    path_corridor_mask,
    pose_region_astar_search,
    hierarchical_pose_region_astar,
    region_astar_path,
)
from substation_vln.planning.improved_astar.visibility import (
    VoxelVisibilityMap,
    build_or_load_voxel_visibility_map,
    candidate_visibility,
)

__all__ = [
    "CameraConfig",
    "VoxelVisibilityMap",
    "PoseAStarConfig",
    "HierarchicalPoseAStarResult",
    "PoseAStarResult",
    "build_pose_free_masks",
    "build_or_load_voxel_visibility_map",
    "candidate_visibility",
    "generate_goal_pose_candidates",
    "pack_pose_free_masks",
    "path_corridor_mask",
    "pose_region_astar_search",
    "hierarchical_pose_region_astar",
    "quantize_heading",
    "region_astar_path",
    "unpack_pose_free_masks",
]
