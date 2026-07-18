"""Coordinate-system conversion helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def z_up_to_habitat_y_up_matrix() -> np.ndarray:
    """Return the rotation matrix mapping Z-up data into Habitat's Y-up frame.

    The mapping keeps X unchanged, maps old Z to new Y, and maps old Y to
    negative new Z:

        (x, y, z) -> (x, z, -y)
    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def transform_z_up_points_to_habitat_y_up(points: np.ndarray) -> np.ndarray:
    """Map Z-up coordinates into Habitat's Y-up coordinate frame."""
    pts = np.asarray(points, dtype=np.float64)
    return pts @ z_up_to_habitat_y_up_matrix().T


def world_camera_to_raw_gaussian_pose(
    gaussian_to_world: np.ndarray,
    position_xyz: np.ndarray,
    yaw_rad: float,
    pitch_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a Z-up world camera pose directly in raw Gaussian coordinates.

    Habitat-GS' current static-Gaussian drawable ignores the scene-node
    transformation supplied to ``RenderInstanceHelper``.  Consequently the
    camera, rather than the drawable, must be transformed into the raw PLY
    frame.  The returned quaternion is the camera-to-world rotation expected
    by Habitat, whose camera looks along local -Z with local +Y as image up.
    """
    transform = np.asarray(gaussian_to_world, dtype=np.float64)
    position = np.asarray(position_xyz, dtype=np.float64)
    if transform.shape != (4, 4) or position.shape != (3,):
        raise ValueError("Expected a 4x4 transform and a 3D camera position")

    inverse_linear = np.linalg.inv(transform[:3, :3])
    raw_position = inverse_linear @ (position - transform[:3, 3])

    cy, sy = np.cos(float(yaw_rad)), np.sin(float(yaw_rad))
    cp, sp = np.cos(float(pitch_rad)), np.sin(float(pitch_rad))
    forward = np.array([cp * cy, cp * sy, sp], dtype=np.float64)
    right = np.array([sy, -cy, 0.0], dtype=np.float64)
    backward = -forward
    up = np.cross(backward, right)
    camera_to_world = np.column_stack([right, up, backward])

    raw_camera_rotation = inverse_linear @ camera_to_world
    # Remove the inverse similarity scale and numerical drift while retaining
    # the proper camera rotation.
    u, _, vt = np.linalg.svd(raw_camera_rotation)
    raw_camera_rotation = u @ vt
    if np.linalg.det(raw_camera_rotation) < 0.0:
        u[:, -1] *= -1.0
        raw_camera_rotation = u @ vt
    return raw_position, rotation_matrix_to_quaternion_wxyz(raw_camera_rotation)


def rotation_matrix_to_quaternion_wxyz(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion in Habitat-GS [w, x, y, z] order."""
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise SystemExit("Gaussian coordinate conversion requires scipy.") from exc

    quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def load_habitat_gs_rotate_module(habitat_gs_root: Path) -> Any:
    """Load Habitat-GS' official Gaussian rotation implementation."""
    rotate_script = habitat_gs_root / "tools_gs" / "rotate_gs.py"
    if not rotate_script.exists():
        raise SystemExit(f"Habitat-GS rotate_gs.py not found: {rotate_script}")

    spec = importlib.util.spec_from_file_location("habitat_gs_rotate_gs", rotate_script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import Habitat-GS rotate script: {rotate_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_gaussian_z_up_to_y_up(input_path: Path, output_path: Path, habitat_gs_root: Path) -> None:
    """Convert a Z-up 3D Gaussian PLY into a Y-up PLY for Habitat-GS viewing.

    This delegates the asset-level work to Habitat-GS' official rotate_gs.py
    implementation so Gaussian centers, normals, quaternions, and SH
    coefficients are rotated consistently.
    """
    rotate_gs = load_habitat_gs_rotate_module(habitat_gs_root)
    rot_matrix = z_up_to_habitat_y_up_matrix()
    rot_quat_wxyz = rotation_matrix_to_quaternion_wxyz(rot_matrix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rotate_gs.rotate_ply(
        str(input_path),
        str(output_path),
        rot_matrix,
        rot_quat_wxyz,
        {},
    )


def ensure_gaussian_z_up_to_y_up_cache(input_path: Path, output_path: Path, habitat_gs_root: Path) -> Path:
    """Return a Y-up Gaussian cache path, creating it when absent or stale."""
    if output_path.exists() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
        return output_path

    convert_gaussian_z_up_to_y_up(input_path, output_path, habitat_gs_root)
    return output_path
