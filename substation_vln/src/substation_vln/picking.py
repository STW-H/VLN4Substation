"""Open3D interactive point picking helpers."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any
import time

import numpy as np

from .interactive import pause, print_section


class SceneWidgetInspectionPicker:
    """Persistent SceneWidget picker for one-point-at-a-time device annotation."""

    RECORDED_MARKER_NAME = "recorded_inspection_points"
    CANDIDATE_MARKER_NAME = "candidate_inspection_point"

    def __init__(
        self,
        o3d: Any,
        pcd,
        *,
        point_size: float = 3.0,
        marker_point_size: float = 8.0,
        background_color: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 1.0),
    ) -> None:
        self.o3d = o3d
        self.pcd = pcd
        self.points = np.asarray(pcd.points)
        if len(self.points) == 0:
            raise ValueError("Cannot pick inspection points from an empty point cloud.")
        print(f"  正在为 {len(self.points):,} 个显示点构建拾取索引...")
        started = time.perf_counter()
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)
        print(f"  拾取索引构建完成，耗时 {time.perf_counter() - started:.2f} s。")

        self.gui = o3d.visualization.gui
        self.rendering = o3d.visualization.rendering
        self.app = self.gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("三维设备巡视点位标注", 1280, 800)
        self.widget = self.gui.SceneWidget()
        self.window.add_child(self.widget)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget.scene = self.rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background(np.asarray(background_color, dtype=np.float32))

        base_material = self.rendering.MaterialRecord()
        base_material.shader = "defaultUnlit"
        base_material.point_size = float(point_size)
        self.marker_material = self.rendering.MaterialRecord()
        self.marker_material.shader = "defaultUnlit"
        self.marker_material.point_size = float(marker_point_size)
        self.widget.scene.add_geometry("pointcloud", pcd, base_material)
        bounds = self.widget.scene.bounding_box
        self.widget.setup_camera(60.0, bounds, bounds.get_center())
        self.widget.set_on_mouse(self._on_mouse)
        self.widget.set_on_key(self._on_key)

        self.recorded_display_points: list[np.ndarray] = []
        self.candidate_display_point: np.ndarray | None = None
        self.action: str | None = None
        self.selected_display_point: np.ndarray | None = None
        self.depth_request_pending = False
        self.pick_request_serial = 0
        self.terminal_task_active = False
        self.closed = False

    def _on_layout(self, _context) -> None:
        self.widget.frame = self.window.content_rect

    def _on_close(self) -> bool:
        self.closed = True
        self.action = "quit"
        return True

    def _update_markers(self) -> None:
        if self.widget.scene.has_geometry(self.RECORDED_MARKER_NAME):
            self.widget.scene.remove_geometry(self.RECORDED_MARKER_NAME)
        if self.widget.scene.has_geometry(self.CANDIDATE_MARKER_NAME):
            self.widget.scene.remove_geometry(self.CANDIDATE_MARKER_NAME)
        if self.recorded_display_points:
            marker = self.o3d.geometry.PointCloud()
            marker.points = self.o3d.utility.Vector3dVector(np.asarray(self.recorded_display_points))
            marker.colors = self.o3d.utility.Vector3dVector(
                np.tile(np.asarray([[1.0, 0.85, 0.0]]), (len(self.recorded_display_points), 1))
            )
            self.widget.scene.add_geometry(self.RECORDED_MARKER_NAME, marker, self.marker_material)
        if self.candidate_display_point is not None:
            candidate = self.o3d.geometry.PointCloud()
            candidate.points = self.o3d.utility.Vector3dVector(
                np.asarray([self.candidate_display_point], dtype=np.float64)
            )
            candidate.colors = self.o3d.utility.Vector3dVector(np.asarray([[1.0, 0.85, 0.0]]))
            self.widget.scene.add_geometry(self.CANDIDATE_MARKER_NAME, candidate, self.marker_material)
        self.widget.force_redraw()

    def start_device(self) -> None:
        self.pick_request_serial += 1
        self.depth_request_pending = False
        self.recorded_display_points = []
        self.candidate_display_point = None
        self._update_markers()

    def commit_selected_point(self, point: np.ndarray) -> None:
        self.recorded_display_points.append(np.asarray(point, dtype=np.float64))
        self.candidate_display_point = None
        self._update_markers()

    def _on_mouse(self, event):
        handled = self.gui.Widget.EventCallbackResult.HANDLED
        ignored = self.gui.Widget.EventCallbackResult.IGNORED
        if self.terminal_task_active:
            return handled
        if event.type != self.gui.MouseEvent.Type.BUTTON_DOWN or not event.is_modifier_down(self.gui.KeyModifier.SHIFT):
            return ignored

        if event.is_button_down(self.gui.MouseButton.RIGHT):
            self.pick_request_serial += 1
            self.depth_request_pending = False
            if self.candidate_display_point is None:
                print("当前没有待确认的候选点。")
                return handled
            self.candidate_display_point = None
            self._update_markers()
            print("已取消当前候选点，请重新选择。")
            return handled

        if not event.is_button_down(self.gui.MouseButton.LEFT):
            return ignored
        if self.depth_request_pending:
            print("正在处理上一次点击，请稍候。")
            return handled

        x = int(event.x - self.widget.frame.x)
        y = int(event.y - self.widget.frame.y)
        width = int(self.widget.frame.width)
        height = int(self.widget.frame.height)
        self.depth_request_pending = True
        self.pick_request_serial += 1
        request_serial = self.pick_request_serial

        def depth_callback(depth_image) -> None:
            depth_array = np.asarray(depth_image)
            selected: np.ndarray | None = None
            if 0 <= x < width and 0 <= y < height:
                depth = float(depth_array[y, x])
                if np.isfinite(depth) and depth < 1.0:
                    world = self.widget.scene.camera.unproject(x, y, depth, width, height)
                    count, indices, _ = self.kdtree.search_knn_vector_3d(np.asarray(world), 1)
                    if count:
                        selected = np.asarray(self.points[int(indices[0])], dtype=np.float64)

            def finish_pick() -> None:
                if request_serial != self.pick_request_serial:
                    return
                self.depth_request_pending = False
                if selected is None:
                    print("当前点击没有命中点云，请重新选择。")
                    return
                self.candidate_display_point = selected
                self._update_markers()
                print("候选点已标黄：按 Enter 确认，或 Shift+右键取消。")

            self.app.post_to_main_thread(self.window, finish_pick)

        self.widget.scene.scene.render_to_depth_image(depth_callback)
        return handled

    def _on_key(self, event):
        handled = self.gui.Widget.EventCallbackResult.HANDLED
        ignored = self.gui.Widget.EventCallbackResult.IGNORED
        if self.terminal_task_active:
            return handled
        if event.type == self.gui.KeyEvent.Type.DOWN:
            if event.key == self.gui.KeyName.ENTER:
                if self.candidate_display_point is None:
                    print("尚未选择候选点：请先 Shift+左键选择。")
                    return handled
                self.selected_display_point = np.asarray(self.candidate_display_point, dtype=np.float64)
                self.action = "confirm_point"
                return handled
            if event.key == self.gui.KeyName.Q:
                self.pick_request_serial += 1
                self.depth_request_pending = False
                if self.candidate_display_point is not None:
                    print("当前未确认候选点已放弃。")
                    self.candidate_display_point = None
                    self._update_markers()
                self.action = "finish_device"
                return handled
        return ignored

    def wait_for_action(self, equipment_name: str) -> tuple[str, np.ndarray | None]:
        if self.closed:
            return "quit", None
        self.action = None
        self.selected_display_point = None
        self.window.set_focus_widget(self.widget)
        print_section(f"标注设备：{equipment_name}", char="-")
        print("Shift+左键：选择候选点；Shift+右键：取消候选；Enter：确认并命名；Q：完成设备。")
        while self.action is None and not self.closed:
            if not self.app.run_one_tick():
                self.closed = True
                self.action = "quit"
                break
        return self.action or "quit", self.selected_display_point

    def run_terminal_task(self, task: Callable[[], Any]) -> Any:
        """Run blocking terminal input while the main thread keeps the GUI responsive."""
        result: list[Any] = []
        error: list[BaseException] = []

        def worker() -> None:
            try:
                result.append(task())
            except BaseException as exc:  # Propagate terminal-input and save errors on the main thread.
                error.append(exc)

        thread = threading.Thread(target=worker, name="inspection-terminal-input", daemon=True)
        self.terminal_task_active = True
        thread.start()
        try:
            while thread.is_alive() and not self.closed:
                if not self.app.run_one_tick():
                    self.closed = True
                    break
            thread.join(timeout=0.1)
        finally:
            self.terminal_task_active = False
        if error:
            raise error[0]
        if self.closed and thread.is_alive():
            raise RuntimeError("标注窗口已关闭，终端输入已中止。")
        return result[0] if result else None

    def close(self) -> None:
        if not self.closed:
            self.window.close()
            self.app.run_one_tick()
            self.closed = True


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


def pick_last_point_with_pause(o3d: Any, pcd, title: str) -> np.ndarray:
    """Allow exploratory picking and return the last point left in the queue."""
    print_section(f"Next window: {title}", char="-")
    print("You may pick several candidates; the last point left in the queue will be used.")
    print("Controls: Shift + left click = pick, Shift + right click = undo last pick, Q = finish.")
    pause("Press Enter here to open this picking window...")

    print_section(title)
    print("Pick one or more candidate points, then press Q.")
    print("The program will use the last point still present in the picking queue.")
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
    if not picked:
        raise SystemExit("No point was picked. Please run again and select at least one point.")
    if len(picked) > 1:
        print(f"Picked {len(picked)} candidate points; using the last one (index {picked[-1]}).")
    points = np.asarray(pcd.points)
    selected = np.asarray(points[int(picked[-1])], dtype=np.float64)
    print(f"Selected index: {picked[-1]}")
    print(f"Selected coordinate: {selected}")
    return selected


def pick_one_point_or_finish_device_with_pause(o3d: Any, pcd, title: str) -> np.ndarray | None:
    """Pick one point, or return None when Q is pressed with an empty queue."""
    print_section(f"Next window: {title}", char="-")
    print("选择一个巡视点位后按 Q，随后在命令行输入名称。")
    print("若当前设备已经完成，请不选择任何点，直接按 Q。")
    print("Controls: Shift + left click = pick, Shift + right click = undo, Q = confirm/close.")
    pause("Press Enter here to open this picking window...")

    print_section(title)
    print("Shift+左键选择一个点并按 Q；若直接按 Q，则完成当前设备。")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=800)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = 4.0
        render_option.background_color = np.asarray([0.02, 0.02, 0.02])
    vis.run()
    vis.destroy_window()

    picked = [int(index) for index in vis.get_picked_points()]
    if not picked:
        print("未选择新点：当前设备巡视点位标注完成。")
        return None
    points = np.asarray(pcd.points)
    if len(picked) > 1:
        print(f"本次选择了 {len(picked)} 个点，只记录最后一个；建议每次只选择一个点。")
    selected_index = picked[-1]
    selected = np.asarray(points[selected_index], dtype=np.float64)
    print(f"Selected index: {selected_index}")
    print(f"Selected coordinate: {selected}")
    return selected
