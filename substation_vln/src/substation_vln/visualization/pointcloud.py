"""Open3D visualization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..preprocessing.pointcloud_io import describe_pcd, make_pcd


DEFAULT_BACKGROUND = np.asarray([0.02, 0.02, 0.02])


def centered_display_pcd(o3d: Any, pcd):
    """Return a centered copy for stable Open3D display and the subtracted center."""
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return make_pcd(o3d, points - center, color=(0.7, 0.7, 0.7), colors=colors), center


def coordinate_frame_for_points(o3d: Any, points: np.ndarray, ratio: float = 0.05, min_size: float = 1.0):
    extent = np.ptp(points, axis=0)
    frame_size = max(float(extent.max()) * ratio, min_size)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)


def configure_visualizer(vis, point_size: float = 2.0, background=DEFAULT_BACKGROUND) -> None:
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = point_size
        render_option.background_color = np.asarray(background, dtype=np.float64)


def configure_default_camera(vis) -> None:
    view_control = vis.get_view_control()
    if view_control is not None:
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_front([0.0, -1.0, 0.35])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(0.65)


def draw_point_cloud(
    o3d: Any,
    pcd,
    title: str,
    point_size: float = 2.0,
    show_frame: bool = True,
    describe: bool = False,
    note: str | None = None,
) -> None:
    if describe:
        print(describe_pcd(pcd))
    if note:
        print(note)

    geometries = [pcd]
    points = np.asarray(pcd.points)
    if show_frame:
        geometries.append(coordinate_frame_for_points(o3d, points))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    for geometry in geometries:
        vis.add_geometry(geometry)
    configure_visualizer(vis, point_size=point_size)
    configure_default_camera(vis)
    vis.run()
    vis.destroy_window()


def crop_point_cloud(o3d: Any, pcd, title: str, point_size: float = 3.0):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pcd)
    configure_visualizer(vis, point_size=point_size)
    vis.run()
    vis.destroy_window()
    return vis.get_cropped_geometry()
