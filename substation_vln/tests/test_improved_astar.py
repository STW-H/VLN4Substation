"""Focused tests for camera-constrained region-goal pose planning."""

from __future__ import annotations

import math
import unittest

import numpy as np

from substation_vln.planning.common.grid import GridSpec
from substation_vln.annotation.schema import split_equipment_pending
from substation_vln.planning.common.base_map import build_base_masks, extract_equipment_regions
from substation_vln.planning.improved_astar.camera_model import CameraConfig
from substation_vln.planning.improved_astar.goal_pose_region import (
    build_pose_free_masks,
    generate_goal_pose_candidates,
)
from substation_vln.planning.improved_astar.pose_region_astar import PoseAStarConfig, pose_region_astar_search
from substation_vln.planning.improved_astar.visibility import VoxelVisibilityMap, candidate_visibility


class PosePlanningTest(unittest.TestCase):
    def test_equipment_index_mask_supports_circles_and_polygons(self):
        grid = GridSpec(0.0, 10.0, 0.0, 10.0, 1.0, 10, 10)
        payload = {
            "annotations": [
                {
                    "category": "planning_boundary",
                    "geometry_type": "multi_polygon",
                    "polygons_xy": [[[0.0, 0.0], [9.9, 0.0], [9.9, 9.9], [0.0, 9.9]]],
                },
                {
                    "category": "equipment_region",
                    "geometry_type": "multi_circle",
                    "equipment_name": "test_breaker",
                    "equipment_type": "duanluqi",
                    "circles": [
                        {
                            "center_xy": [2.0, 2.0],
                            "radius_xy": 0.5,
                            "bbox_xy": {"min": [1.5, 1.5], "max": [2.5, 2.5]},
                            "area_xy": math.pi * 0.25,
                        },
                        {
                            "center_xy": [4.0, 2.0],
                            "radius_xy": 0.5,
                            "bbox_xy": {"min": [3.5, 1.5], "max": [4.5, 2.5]},
                            "area_xy": math.pi * 0.25,
                        },
                    ],
                },
                {
                    "category": "equipment_region",
                    "geometry_type": "multi_polygon",
                    "equipment_name": "test_transformer",
                    "polygons_xy": [[[6.0, 6.0], [8.0, 6.0], [8.0, 8.0], [6.0, 8.0]]],
                },
            ]
        }
        layers = build_base_masks(payload, grid, preferred_path_width_m=0.5)
        labels = set(np.unique(layers["equipment_index_mask"]).tolist())
        self.assertTrue({1, 2, 3}.issubset(labels))
        equipment = extract_equipment_regions(payload)
        self.assertEqual(
            [item["equipment_name"] for item in equipment],
            ["test_breaker_1", "test_breaker_2", "test_transformer"],
        )
        self.assertEqual([item["equipment_index"] for item in equipment], [1, 2, 3])

    def test_equipment_circle_batch_is_split_before_saving(self):
        pending = {
            "selection_type": "image_multi_circle",
            "geometry_type": "multi_circle",
            "circles_pixel": [
                {"center_pixel": [10.0, 20.0], "radius_pixel": 4.0},
                {"center_pixel": [30.0, 40.0], "radius_pixel": 5.0},
            ],
        }
        split = split_equipment_pending(pending)
        self.assertEqual(len(split), 2)
        self.assertEqual([len(item["circles_pixel"]) for item in split], [1, 1])
        self.assertEqual(split[0]["circles_pixel"][0]["center_pixel"], [10.0, 20.0])
        self.assertEqual(split[1]["circles_pixel"][0]["center_pixel"], [30.0, 40.0])

    def test_roi_conical_approach_inverts_tilt_to_distance(self):
        grid = GridSpec(0.0, 10.0, 0.0, 10.0, 0.1, 100, 100)
        pose_free = np.ones((8, 100, 100), dtype=np.uint8)
        equipment = {
            "equipment_name": "test_device",
            "equipment_type": "default",
            "center_xyz": [5.0, 5.0, 2.5],
            "robust_bounds_min_xyz": [4.5, 4.5, 0.2],
            "robust_bounds_max_xyz": [5.5, 5.5, 5.0],
            "bbox_xy": {"min": [4.5, 4.5], "max": [5.5, 5.5]},
            "polygons_xy": [[[4.5, 4.5], [5.5, 4.5], [5.5, 5.5], [4.5, 5.5]]],
        }
        camera = CameraConfig(tilt_min_deg=20.0, tilt_max_deg=70.0, preferred_tilt_deg=45.0)
        result = generate_goal_pose_candidates(
            equipment,
            pose_free,
            grid,
            camera,
            {
                "observation_model": "roi_conical_approach",
                "candidate_stride_cells": 2,
                "min_candidate_distance_m": 0.2,
                "max_search_radius_m": 30.0,
            },
            {
                "default": {
                    "vertical_min_fraction": 0.5,
                    "vertical_max_fraction": 1.0,
                    "tilt_min_deg": 30.0,
                    "tilt_max_deg": 70.0,
                    "preferred_tilt_deg": 45.0,
                }
            },
        )
        self.assertGreater(len(result["rows"]), 0)
        index = 0
        x, y = grid.grid_to_xy(
            np.asarray([result["cols"][index]]), np.asarray([result["rows"][index]])
        )
        distance = math.hypot(float(x[0] - 5.0), float(y[0] - 5.0))
        roi_center_z = 0.5 * (2.6 + 5.0)
        expected_tilt = math.atan2(roi_center_z - camera.height_m, distance)
        self.assertAlmostEqual(float(result["camera_tilt_rad"][index]), expected_tilt, places=5)

    def test_rotated_rectangle_and_region_goal_search(self):
        grid = GridSpec(0.0, 2.0, 0.0, 2.0, 0.1, 20, 20)
        boundary = np.ones((20, 20), dtype=np.uint8)
        obstacle = np.zeros_like(boundary)
        equipment = np.zeros_like(boundary)
        obstacle[8:12, 9:11] = 1
        masks = build_pose_free_masks(
            boundary,
            obstacle,
            equipment,
            grid,
            {"length_m": 0.4, "width_m": 0.2, "safety_margin_m": 0.0, "heading_bins": 8},
        )
        self.assertEqual(masks.shape, (8, 20, 20))
        self.assertEqual(int(masks[0, 10, 10]), 0)

        cost = np.ones((20, 20), dtype=np.float32)
        goals = {(4, 15, 2): 0.0, (4, 15, 6): 0.5}
        result = pose_region_astar_search(
            masks,
            cost,
            (15, 4, 0),
            goals,
            PoseAStarConfig(),
            resolution_m=grid.resolution_m,
        )
        self.assertTrue(result.found)
        self.assertIn(result.path_states[-1], goals)

    def test_visibility_checks_only_the_local_ray_corridor(self):
        origins = np.asarray([[1.0, 5.0, 1.0]])
        target = np.asarray([9.0, 5.0, 1.0])
        config = {
            "clearance_radius_m": 0.2,
            "include_voxel_uncertainty": False,
            "ray_step_m": 0.1,
            "camera_exclusion_m": 0.2,
            "target_exclusion_m": 0.2,
            "batch_size": 8,
        }
        blocked_map = VoxelVisibilityMap(np.asarray([[5.0, 5.0, 1.0]], dtype=np.float32), 0.2)
        feasible = candidate_visibility(origins, target, blocked_map, config)
        self.assertFalse(bool(feasible[0]))

        clear_map = VoxelVisibilityMap(np.asarray([[5.0, 7.0, 1.0]], dtype=np.float32), 0.2)
        feasible = candidate_visibility(origins, target, clear_map, config)
        self.assertTrue(bool(feasible[0]))


if __name__ == "__main__":
    unittest.main()
