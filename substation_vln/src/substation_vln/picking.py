"""Open3D interactive point picking helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .interactive import pause, print_section


def pick_points(o3d: Any, pcd, title: str, num_points: int) -> np.ndarray:
    print_section(title)
    print(f"Please pick exactly {num_points} points.")
    print("Open3D controls: Shift + left click to pick, Shift + right click to undo, Q to finish.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = 4.0
        render_option.background_color = np.asarray([0.02, 0.02, 0.02])
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if len(picked) != num_points:
        raise SystemExit(f"Expected {num_points} picked points, got {len(picked)}")
    points = np.asarray(pcd.points)
    picked_points = points[np.asarray(picked, dtype=np.int64)]
    print(f"Picked indices: {picked}")
    print(f"Picked coordinates:\n{picked_points}")
    return picked_points


def pick_with_pause(o3d: Any, pcd, title: str, num_points: int) -> np.ndarray:
    print_section(f"Next window: {title}", char="-")
    print(f"Pick exactly {num_points} points, then press Q in the Open3D window.")
    print("Controls: Shift + left click = pick, Shift + right click = undo.")
    pause("Press Enter here to open this picking window...")
    return pick_points(o3d, pcd, title, num_points)
