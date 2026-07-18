#!/usr/bin/env python3
"""Play a planned route as a fixed-speed first-person Habitat-GS camera."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "substation_vln" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from substation_vln.config import load_yaml_config  # noqa: E402
from substation_vln.paths import CONFIGS_DIR  # noqa: E402
from substation_vln.planning.common.io import read_json, resolve_project_path  # noqa: E402
from substation_vln.preprocessing.coordinate_transforms import (  # noqa: E402
    world_camera_to_raw_gaussian_pose,
)
from substation_vln.visualization.habitat_gs import (  # noqa: E402
    HabitatGSTrajectoryRenderer,
)
from substation_vln.visualization.trajectory_player import (  # noqa: E402
    trajectory_from_route_payload,
)


DEFAULT_CONFIG = (
    CONFIGS_DIR / "tools" / "visualization" / "play_planned_route_erfeishan.yaml"
)


def latest_route(route_dir: Path) -> Path:
    candidates = sorted(
        route_dir.glob("natural_language_route_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No planned route JSON found in {route_dir}")
    return candidates[0]


def ensure_gaussian_stage_link(source: Path, link: Path) -> Path:
    """Give the raw Gaussian the suffix Habitat-GS uses for stage detection."""
    source = source.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == source:
        return link
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(source)
    return link


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play the latest planned route in first-person Habitat-GS view."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--route", type=Path, help="Override paths.route_json")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    paths = config["paths"]

    if args.route:
        route_path = args.route.expanduser().resolve()
    elif paths.get("route_json"):
        route_path = resolve_project_path(paths["route_json"])
    else:
        route_path = latest_route(resolve_project_path(paths["route_dir"]))
    route_payload = read_json(route_path)
    mode = str(
        route_payload.get("instruction_plan", {}).get("movement_mode", "unknown")
    )

    gaussian = resolve_project_path(paths["gaussian"])
    registration = read_json(resolve_project_path(paths["registration_json"]))
    transform = np.asarray(registration["final_matrix"], dtype=np.float64)
    stage_link = ensure_gaussian_stage_link(
        gaussian, resolve_project_path(paths["gaussian_stage_link"])
    )

    playback = config["playback"]
    camera = config["camera"]
    display = config["display"]
    trajectory = trajectory_from_route_payload(
        route_payload,
        camera_height_m=float(camera["height_m"]),
        linear_speed_mps=float(playback["linear_speed_mps"]),
        angular_speed_deg_s=float(playback["angular_speed_deg_s"]),
        apply_terminal_camera_pose=bool(
            playback.get("apply_terminal_camera_pose", True)
        ),
        terminal_hold_s=float(playback.get("terminal_hold_s", 2.0)),
        travel_yaw_source=str(playback.get("travel_yaw_source", "motion_tangent")),
    )

    fps = float(playback.get("fps", 24.0))
    if fps <= 0.0:
        raise SystemExit("playback.fps must be positive")
    loop = bool(playback.get("loop", False))
    window_name = str(display.get("window_name", "Habitat-GS planned route"))
    width = int(camera.get("width", 1280))
    height = int(camera.get("height", 720))

    print(f"轨迹：{route_path}")
    print(f"模式：{mode}")
    print(f"路径播放时长：{trajectory.duration_s:.2f} s")
    print(f"线速度：{float(playback['linear_speed_mps']):.2f} m/s")
    print("坐标方式：相机位姿反变换到原始 Gaussian 坐标（静态实例变换兼容模式）")
    print("控制：Space 暂停/继续，R 重新播放，Q/Esc 退出")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    playback_time = 0.0
    previous_clock = time.monotonic()
    paused = False

    with HabitatGSTrajectoryRenderer(
        stage_link,
        width=width,
        height=height,
        hfov_deg=float(camera.get("hfov_deg", 90.0)),
        instance_translation=[0.0, 0.0, 0.0],
        instance_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        instance_scale=1.0,
        gpu_device_id=int(camera.get("gpu_device_id", 0)),
    ) as renderer:
        while True:
            frame_start = time.monotonic()
            elapsed = frame_start - previous_clock
            previous_clock = frame_start
            if not paused:
                playback_time += elapsed
            if playback_time >= trajectory.duration_s:
                if loop and trajectory.duration_s > 0.0:
                    playback_time %= trajectory.duration_s
                else:
                    playback_time = trajectory.duration_s
                    paused = True

            pose = trajectory.pose_at(playback_time)
            gaussian_position, gaussian_rotation = world_camera_to_raw_gaussian_pose(
                transform,
                np.asarray([pose.x, pose.y, pose.z], dtype=np.float64),
                pose.yaw_rad,
                pose.pitch_rad,
            )
            frame_rgb = renderer.render(
                gaussian_position.tolist(),
                rotation_wxyz=gaussian_rotation.tolist(),
            )
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame,
                f"{mode}  {playback_time:6.1f}/{trajectory.duration_s:6.1f}s",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            remaining = max(0.0, 1.0 / fps - (time.monotonic() - frame_start))
            key = cv2.waitKey(max(1, int(round(remaining * 1000.0)))) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
                previous_clock = time.monotonic()
            elif key in (ord("r"), ord("R")):
                playback_time = 0.0
                paused = False
                previous_clock = time.monotonic()
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
