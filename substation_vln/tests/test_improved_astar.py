"""Focused tests for camera-constrained region-goal pose planning."""

from __future__ import annotations

import math
import unittest

import numpy as np

from substation_vln.planning.common.grid import GridSpec
from substation_vln.annotation.schema import CATEGORIES, make_annotation, split_equipment_pending
from substation_vln.planning.common.base_map import (
    build_base_masks,
    extract_equipment_regions,
    extract_robot_start_points,
)
from substation_vln.planning.improved_astar.camera_model import CameraConfig
from substation_vln.planning.improved_astar.goal_pose_region import (
    build_pose_free_masks,
    generate_goal_pose_candidates,
)
from substation_vln.planning.improved_astar.pose_region_astar import (
    PoseAStarConfig,
    hierarchical_pose_region_astar,
    pose_region_astar_search,
)
from substation_vln.planning.improved_astar.visibility import VoxelVisibilityMap, candidate_visibility
from substation_vln.tasks.schema import RoutePlan
from substation_vln.tasks.instruction_parser import (
    canonicalize_catalog_references,
)
from substation_vln.preprocessing.coordinate_transforms import (
    world_camera_to_raw_gaussian_pose,
)
from substation_vln.visualization.trajectory_player import trajectory_from_route_payload


class PosePlanningTest(unittest.TestCase):
    def test_world_camera_pose_is_mapped_into_untransformed_gaussian(self):
        angle = math.radians(30.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = np.eye(4)
        transform[:3, :3] = 2.0 * rotation
        transform[:3, 3] = [100.0, 200.0, 3.0]
        expected_raw_position = np.asarray([4.0, -2.0, 1.5])
        world_position = (
            transform[:3, :3] @ expected_raw_position + transform[:3, 3]
        )
        yaw, pitch = 0.7, 0.2

        raw_position, quat = world_camera_to_raw_gaussian_pose(
            transform, world_position, yaw, pitch
        )
        np.testing.assert_allclose(raw_position, expected_raw_position, atol=1.0e-9)

        w, x, y, z = quat
        raw_camera_rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        raw_forward = raw_camera_rotation @ np.asarray([0.0, 0.0, -1.0])
        world_forward = transform[:3, :3] @ raw_forward
        world_forward /= np.linalg.norm(world_forward)
        expected_forward = np.asarray(
            [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
        )
        np.testing.assert_allclose(world_forward, expected_forward, atol=1.0e-9)

    def test_camera_trajectory_uses_fixed_linear_and_angular_speeds(self):
        payload = {
            "route": {
                "states": [
                    {"xy": [0.0, 0.0], "yaw_rad": 0.0},
                    {"xy": [1.0, 0.0], "yaw_rad": 0.0},
                    {"xy": [1.0, 0.0], "yaw_rad": math.pi / 2.0},
                ]
            }
        }
        trajectory = trajectory_from_route_payload(
            payload,
            camera_height_m=1.0,
            linear_speed_mps=0.5,
            angular_speed_deg_s=45.0,
            apply_terminal_camera_pose=False,
            terminal_hold_s=0.0,
        )
        self.assertAlmostEqual(trajectory.duration_s, 4.0)
        self.assertAlmostEqual(trajectory.pose_at(1.0).x, 0.5)
        self.assertAlmostEqual(trajectory.pose_at(3.0).yaw_rad, math.pi / 4.0)

    def test_terminal_camera_keeps_positive_z_up_elevation_in_habitat(self):
        payload = {
            "route": {"states": [{"xy": [2.0, 3.0], "yaw_rad": 0.2}]},
            "target": {
                "route_segment": {
                    "camera": {"pan_rad": 0.3, "tilt_rad": 0.4}
                }
            },
        }
        trajectory = trajectory_from_route_payload(
            payload,
            camera_height_m=1.0,
            linear_speed_mps=0.5,
            angular_speed_deg_s=45.0,
            apply_terminal_camera_pose=True,
            terminal_hold_s=0.0,
        )
        final_pose = trajectory.pose_at(trajectory.duration_s)
        self.assertAlmostEqual(final_pose.yaw_rad, 0.5)
        self.assertAlmostEqual(final_pose.pitch_rad, 0.4)

    def test_motion_tangent_camera_yaw_follows_xy_trajectory(self):
        payload = {
            "route": {
                "states": [
                    {"xy": [0.0, 0.0], "yaw_rad": 0.0},
                    {"xy": [0.0, 1.0], "yaw_rad": 0.0},
                ]
            }
        }
        trajectory = trajectory_from_route_payload(
            payload,
            camera_height_m=1.0,
            linear_speed_mps=1.0,
            angular_speed_deg_s=45.0,
            apply_terminal_camera_pose=False,
            terminal_hold_s=0.0,
            travel_yaw_source="motion_tangent",
        )
        self.assertAlmostEqual(trajectory.pose_at(0.0).yaw_rad, math.pi / 2.0)

    def test_robot_start_point_batch_has_selectable_names(self):
        category = next(item for item in CATEGORIES.values() if item["key"] == "robot_start_point")
        annotation = make_annotation(
            annotation_id=1,
            category=category,
            label="gate",
            pending={
                "selection_type": "image_multi_point",
                "geometry_type": "multi_point",
                "points_pixel": [[1.0, 2.0], [3.0, 4.0]],
            },
            pixel_to_world=np.eye(3, dtype=np.float64),
            color_bgr=(1, 2, 3),
        )
        starts = extract_robot_start_points({"annotations": [annotation]})
        self.assertEqual([item["start_point_name"] for item in starts], ["gate_1", "gate_2"])
        self.assertEqual([item["xy"] for item in starts], [[1.0, 2.0], [3.0, 4.0]])

    def test_route_plan_validates_mode_and_intermediate_points(self):
        plan = RoutePlan.from_model_response(
            {
                "start_point": "gate_1",
                "movement_mode": "safe",
                "target_point": "1#duanluqi_1",
                "intermediate_points": ["gate_2"],
            },
            raw_instruction="从gate_1经gate_2安全巡视断路器",
            provider="deepseek",
            model="deepseek-v4-pro",
        )
        self.assertEqual(plan.movement_mode, "safe")
        self.assertEqual(plan.start_point, "gate_1")
        self.assertEqual(plan.intermediate_points, ["gate_2"])

    def test_deepseek_catalog_references_are_canonicalized(self):
        payload = canonicalize_catalog_references(
            {
                "start_point": "ＧＡＴＥ－１",
                "target_point": "1",
                "intermediate_points": ["2"],
            },
            [
                {"start_point_index": 1, "start_point_name": "gate_1"},
                {"start_point_index": 2, "start_point_name": "gate_2"},
            ],
            [{"equipment_index": 1, "equipment_name": "1#duanluqi_1"}],
        )
        self.assertEqual(payload["start_point"], "gate_1")
        self.assertEqual(payload["target_point"], "1#duanluqi_1")
        self.assertEqual(payload["intermediate_points"], ["gate_2"])
        plan = RoutePlan.from_model_response(
            {**payload, "movement_mode": "safe"},
            raw_instruction="test",
            provider="deepseek",
            model="deepseek-v4-pro",
        )
        self.assertEqual(plan.target_point, "1#duanluqi_1")
        self.assertEqual(plan.intermediate_points, ["gate_2"])

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

    def test_only_directed_paths_create_local_direction_vectors(self):
        grid = GridSpec(0.0, 10.0, 0.0, 10.0, 1.0, 10, 10)
        payload = {
            "annotations": [
                {
                    "category": "planning_boundary",
                    "geometry_type": "multi_polygon",
                    "polygons_xy": [
                        [[0.0, 0.0], [9.9, 0.0], [9.9, 9.9], [0.0, 9.9]]
                    ],
                },
                {
                    "category": "preferred_path",
                    "geometry_type": "directed_polyline",
                    "polyline_xy": [[1.0, 5.0], [8.0, 5.0]],
                },
                {
                    "category": "preferred_path",
                    "geometry_type": "polyline",
                    "polyline_xy": [[1.0, 7.0], [8.0, 7.0]],
                },
            ]
        }
        layers = build_base_masks(payload, grid, preferred_path_width_m=1.0)
        directed_col, directed_row = grid.xy_to_grid(
            np.asarray([[4.0, 5.0]], dtype=np.float64)
        )[0]
        undirected_col, undirected_row = grid.xy_to_grid(
            np.asarray([[4.0, 7.0]], dtype=np.float64)
        )[0]
        self.assertAlmostEqual(
            float(layers["preferred_path_direction_x"][directed_row, directed_col]),
            1.0,
        )
        self.assertEqual(
            int(layers["directed_preferred_path_mask"][undirected_row, undirected_col]),
            0,
        )

    def test_directed_path_rewards_only_forward_motion(self):
        pose_free = np.ones((4, 1, 3), dtype=np.uint8)
        cost_map = np.zeros((1, 3), dtype=np.float32)
        direction = (
            np.ones((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        config = PoseAStarConfig(
            cost_weight=0.0,
            lateral_motion_weight=0.0,
            preferred_path_direction_reward=0.4,
            preferred_path_reverse_penalty=0.5,
            allow_diagonal=False,
        )
        forward = pose_region_astar_search(
            pose_free,
            cost_map,
            (0, 0, 0),
            {(0, 2, 0): 0.0},
            config,
            preferred_path_direction=direction,
        )
        reverse = pose_region_astar_search(
            pose_free,
            cost_map,
            (0, 2, 0),
            {(0, 0, 0): 0.0},
            config,
            preferred_path_direction=direction,
        )
        undirected = pose_region_astar_search(
            pose_free,
            cost_map,
            (0, 0, 0),
            {(0, 2, 0): 0.0},
            config,
            preferred_path_direction=(
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
            ),
        )
        self.assertTrue(forward.found and undirected.found and reverse.found)
        self.assertLess(forward.path_cost, undirected.path_cost)
        self.assertLess(undirected.path_cost, reverse.path_cost)

    def test_pose_astar_penalizes_large_path_turns(self):
        pose_free = np.ones((4, 2, 2), dtype=np.uint8)
        cost_map = np.zeros((2, 2), dtype=np.float32)
        no_turn_cost = pose_region_astar_search(
            pose_free,
            cost_map,
            (1, 0, 0),
            {(0, 1, 0): 0.0},
            PoseAStarConfig(
                cost_weight=0.0,
                lateral_motion_weight=0.0,
                path_turn_cost_weight=0.0,
                allow_diagonal=False,
            ),
        )
        with_turn_cost = pose_region_astar_search(
            pose_free,
            cost_map,
            (1, 0, 0),
            {(0, 1, 0): 0.0},
            PoseAStarConfig(
                cost_weight=0.0,
                lateral_motion_weight=0.0,
                path_turn_cost_weight=0.8,
                allow_diagonal=False,
            ),
        )
        self.assertTrue(no_turn_cost.found and with_turn_cost.found)
        self.assertAlmostEqual(
            with_turn_cost.path_cost - no_turn_cost.path_cost,
            0.8,
            places=6,
        )

    def test_pose_astar_replaces_right_angle_with_gradual_turn(self):
        result = pose_region_astar_search(
            np.ones((4, 3, 3), dtype=np.uint8),
            np.zeros((3, 3), dtype=np.float32),
            (2, 0, 0),
            {(0, 1, 0): 0.0},
            PoseAStarConfig(
                cost_weight=0.0,
                lateral_motion_weight=0.0,
                path_turn_cost_weight=0.2,
                max_path_turn_deg=45.0,
                allow_diagonal=True,
            ),
        )
        self.assertTrue(result.found)
        movements = []
        for previous, current in zip(
            result.path_states[:-1], result.path_states[1:], strict=True
        ):
            delta = (current[0] - previous[0], current[1] - previous[1])
            if delta != (0, 0):
                movements.append(delta)
        for previous, current in zip(movements[:-1], movements[1:], strict=True):
            cosine = np.dot(previous, current) / (
                np.linalg.norm(previous) * np.linalg.norm(current)
            )
            angle_deg = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            self.assertLessEqual(angle_deg, 45.0 + 1.0e-6)

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

        hierarchical = hierarchical_pose_region_astar(
            masks,
            cost,
            (15, 4, 0),
            goals,
            PoseAStarConfig(),
            resolution_m=grid.resolution_m,
            corridor_radius_m=0.3,
            max_corridor_radius_m=1.2,
        )
        self.assertTrue(hierarchical.pose_result.found)
        self.assertGreater(len(hierarchical.coarse_path_rc), 0)

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
