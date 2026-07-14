"""Interactive annotation of 3D inspection targets on a processed point cloud."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from substation_vln.interactive import ask_yes_no
from substation_vln.picking import SceneWidgetInspectionPicker
from substation_vln.preprocessing.pointcloud_io import import_open3d, make_pcd, sample_ply_points
from substation_vln.visualization.pointcloud import (
    centered_display_pcd,
    configure_default_camera,
    configure_visualizer,
    coordinate_frame_for_points,
)


SCHEMA_VERSION = 3
INSPECTION_TARGET_CATEGORY = {
    "key": "inspection_target",
    "name": "三维巡视目标",
    "default_label": "inspection_target",
    "geometry": "point_3d",
}


@dataclass(frozen=True)
class InspectionTargetDefaults:
    category: str = "inspection_target"
    task_type: str = "visual_inspection"
    min_observation_distance_m: float = 2.0
    max_observation_distance_m: float = 6.0
    target_exclusion_radius_m: float = 0.2

    def validate(self) -> None:
        validate_observation_parameters(
            self.min_observation_distance_m,
            self.max_observation_distance_m,
            self.target_exclusion_radius_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "task_type": self.task_type,
            "min_observation_distance_m": self.min_observation_distance_m,
            "max_observation_distance_m": self.max_observation_distance_m,
            "target_exclusion_radius_m": self.target_exclusion_radius_m,
        }


def validate_observation_parameters(min_distance_m: float, max_distance_m: float, exclusion_radius_m: float) -> None:
    values = np.asarray([min_distance_m, max_distance_m, exclusion_radius_m], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Observation distances and target exclusion radius must be finite.")
    if min_distance_m < 0:
        raise ValueError("Minimum observation distance must be non-negative.")
    if max_distance_m <= min_distance_m:
        raise ValueError("Maximum observation distance must be greater than minimum distance.")
    if exclusion_radius_m < 0:
        raise ValueError("Target exclusion radius must be non-negative.")
    if exclusion_radius_m >= max_distance_m:
        raise ValueError("Target exclusion radius must be smaller than maximum observation distance.")


def make_target_record(
    *,
    target_id: str,
    label: str,
    category: str,
    task_type: str,
    target_xyz: np.ndarray | list[float],
    ground_z_m: float,
    min_observation_distance_m: float,
    max_observation_distance_m: float,
    target_exclusion_radius_m: float,
    notes: str = "",
    annotation_id: int | None = None,
    source_pointcloud: Path | None = None,
    camera_height_m: float | None = None,
    equipment_id: str | None = None,
    equipment_name: str | None = None,
    equipment_type: str | None = None,
    inspection_point_id: str | None = None,
    inspection_point_name: str | None = None,
) -> dict[str, Any]:
    target_id = target_id.strip()
    if not target_id:
        raise ValueError("target_id cannot be empty.")
    xyz = np.asarray(target_xyz, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("target_xyz must contain exactly three finite coordinates.")
    validate_observation_parameters(
        min_observation_distance_m,
        max_observation_distance_m,
        target_exclusion_radius_m,
    )
    record = {
        "target_id": target_id,
        "label": label.strip() or target_id,
        "category": "inspection_target",
        "category_name": "三维巡视目标",
        "device_category": category.strip() or "inspection_target",
        "selection_type": "pointcloud_point",
        "geometry_type": "point_3d",
        "color_bgr": [30, 30, 255],
        "task_type": task_type.strip() or "visual_inspection",
        "target_xyz": xyz.tolist(),
        "target_height_above_ground_m": float(xyz[2] - ground_z_m),
        "min_observation_distance_m": float(min_observation_distance_m),
        "max_observation_distance_m": float(max_observation_distance_m),
        "target_exclusion_radius_m": float(target_exclusion_radius_m),
        "notes": notes.strip(),
    }
    if annotation_id is not None:
        record["id"] = int(annotation_id)
    if source_pointcloud is not None:
        record["source_pointcloud"] = str(source_pointcloud.expanduser().resolve())
    record["coordinate_system"] = "axis_corrected_pointcloud_z_up"
    record["ground_z_m"] = float(ground_z_m)
    if camera_height_m is not None:
        record["camera_height_m"] = float(camera_height_m)
    if equipment_id is not None:
        record["equipment_id"] = equipment_id.strip()
    if equipment_name is not None:
        record["equipment_name"] = equipment_name.strip()
    if equipment_type is not None:
        record["equipment_type"] = equipment_type.strip()
    if inspection_point_id is not None:
        record["inspection_point_id"] = inspection_point_id.strip()
    if inspection_point_name is not None:
        record["inspection_point_name"] = inspection_point_name.strip()
    return record


def new_annotation_payload(
    *,
    source_pointcloud: Path,
    ground_z_m: float,
    camera_height_m: float,
    max_display_points: int,
    loaded_display_points: int,
    display_center: np.ndarray,
    defaults: InspectionTargetDefaults,
) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_kind": "pointcloud_3d",
        "saved_at": now,
        "source_pointcloud": str(source_pointcloud.expanduser().resolve()),
        "coordinate_frame": "axis_corrected_pointcloud_z_up",
        "ground_plane": {"model": "constant_z", "z_m": float(ground_z_m)},
        "camera": {"model": "omnidirectional", "height_above_ground_m": float(camera_height_m)},
        "display_sampling": {
            "max_points": int(max_display_points),
            "loaded_points": int(loaded_display_points),
            "display_center_subtracted": np.asarray(display_center, dtype=np.float64).tolist(),
            "note": "Display centering and sampling affect only interactive picking; target_xyz is saved in full point-cloud coordinates.",
        },
        "categories": {"7": INSPECTION_TARGET_CATEGORY},
        "default_observation_parameters": defaults.to_dict(),
        "equipment_annotation_policy": "annotate_all_inspection_points_for_one_physical_device_before_starting_the_next_device",
        "annotations": [],
    }


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("annotation_kind") == "pointcloud_3d":
        targets = payload.get("annotations")
    elif payload.get("type") in ("inspection_targets_3d", "pointcloud_inspection_targets"):
        targets = payload.get("targets")
    else:
        raise ValueError("Not a point-cloud inspection-target annotation file.")
    if not isinstance(targets, list):
        raise ValueError("Annotation payload has no targets list.")
    ids = [str(item.get("target_id", "")) for item in targets]
    if any(not item for item in ids):
        raise ValueError("Every target must have a non-empty target_id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Target IDs must be unique.")
    if int(payload.get("schema_version", 0)) >= 3:
        required = ("equipment_id", "equipment_name", "equipment_type", "inspection_point_id", "inspection_point_name")
        for target in targets:
            missing = [key for key in required if not str(target.get(key, "")).strip()]
            if missing:
                raise ValueError(f"3D inspection target {target.get('target_id')} is missing: {missing}")


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    validate_payload(payload)
    timestamp = datetime.now().isoformat(timespec="seconds")
    if payload.get("annotation_kind") == "pointcloud_3d":
        payload["saved_at"] = timestamp
    else:
        payload["updated_at"] = timestamp
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prompt_float(prompt: str, default: float, *, minimum: float = 0.0) -> float:
    while True:
        raw = input(f"{prompt} [{default:g}]: ").strip()
        if not raw:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("请输入有效数字。")
            continue
        if not np.isfinite(value) or value < minimum:
            print(f"请输入不小于 {minimum:g} 的有限数值。")
            continue
        return value


class InspectionTarget3DAnnotator:
    """Orchestrate repeated 3D target picking while preserving full coordinates."""

    def __init__(
        self,
        *,
        pointcloud_path: Path,
        output_path: Path,
        max_display_points: int,
        point_size: float,
        ground_z_m: float,
        camera_height_m: float,
        defaults: InspectionTargetDefaults,
        resume: bool = True,
        show_review: bool = True,
        review_sphere_radius_m: float = 0.3,
        selection_marker_point_size: float = 8.0,
        display_color_contrast: float = 1.0,
        display_color_brightness: float = 0.0,
        display_background_color: tuple[float, float, float, float] = (0.02, 0.02, 0.02, 1.0),
        prompt_target_id: bool = False,
        prompt_task_type: bool = False,
        prompt_observation_parameters: bool = False,
    ) -> None:
        self.pointcloud_path = pointcloud_path.expanduser().resolve()
        self.output_path = output_path.expanduser().resolve()
        self.max_display_points = int(max_display_points)
        self.point_size = float(point_size)
        self.ground_z_m = float(ground_z_m)
        self.camera_height_m = float(camera_height_m)
        self.defaults = defaults
        self.resume = bool(resume)
        self.show_review = bool(show_review)
        self.review_sphere_radius_m = float(review_sphere_radius_m)
        self.selection_marker_point_size = float(selection_marker_point_size)
        self.display_color_contrast = float(display_color_contrast)
        self.display_color_brightness = float(display_color_brightness)
        self.display_background_color = tuple(float(value) for value in display_background_color)
        self.prompt_target_id = bool(prompt_target_id)
        self.prompt_task_type = bool(prompt_task_type)
        self.prompt_observation_parameters = bool(prompt_observation_parameters)
        self.defaults.validate()
        if not self.pointcloud_path.exists():
            raise SystemExit(f"Point cloud not found: {self.pointcloud_path}")
        if self.camera_height_m <= 0:
            raise ValueError("Camera height must be positive.")
        if self.max_display_points < 0:
            raise ValueError("max_display_points must be non-negative; 0 means all points.")
        if min(self.point_size, self.review_sphere_radius_m, self.selection_marker_point_size) <= 0:
            raise ValueError("Point size, review sphere radius, and selection marker point size must be positive.")
        if not np.isfinite(self.display_color_contrast) or self.display_color_contrast <= 0:
            raise ValueError("Display color contrast must be a positive finite number.")
        if not np.isfinite(self.display_color_brightness):
            raise ValueError("Display color brightness must be finite.")
        if len(self.display_background_color) != 4 or not np.all(
            np.isfinite(self.display_background_color)
        ) or not np.all((np.asarray(self.display_background_color) >= 0) & (np.asarray(self.display_background_color) <= 1)):
            raise ValueError("Display background color must contain four values in [0, 1].")

        self.o3d = import_open3d()
        self.display_pcd = None
        self.display_center = np.zeros(3, dtype=np.float64)
        self.payload: dict[str, Any] = {}

    @property
    def targets(self) -> list[dict[str, Any]]:
        if self.payload.get("annotation_kind") == "pointcloud_3d":
            return self.payload["annotations"]
        return self.payload["targets"]

    def load_pointcloud(self) -> None:
        print("\n[三维目标标注 1/3] 读取轴矫正后的完整点云")
        print(f"  输入：{self.pointcloud_path}")
        print(f"  最大显示点数：{self.max_display_points:,}（0表示全部读取）")
        points, colors = sample_ply_points(self.pointcloud_path, self.max_display_points)
        source_pcd = make_pcd(self.o3d, points, colors=colors)
        self.display_pcd, self.display_center = centered_display_pcd(self.o3d, source_pcd)
        if self.display_pcd.has_colors() and (
            not np.isclose(self.display_color_contrast, 1.0)
            or not np.isclose(self.display_color_brightness, 0.0)
        ):
            colors = np.asarray(self.display_pcd.colors)
            chunk_size = 1_000_000
            for start in range(0, len(colors), chunk_size):
                chunk = colors[start : start + chunk_size]
                chunk[:] = np.clip(
                    (chunk - 0.5) * self.display_color_contrast
                    + 0.5
                    + self.display_color_brightness,
                    0.0,
                    1.0,
                )
        print(f"  已载入显示点：{len(points):,}")
        print(f"  显示中心平移：{self.display_center}")
        print(
            "  显示增强："
            f"对比度 {self.display_color_contrast:g}，"
            f"亮度偏移 {self.display_color_brightness:+g}，"
            f"点大小 {self.point_size:g} px"
        )
        print("  注意：中心化只用于Open3D稳定显示，保存的target_xyz会恢复为完整工程坐标。")

    def prepare_payload(self) -> None:
        if self.output_path.exists():
            if not self.resume:
                raise SystemExit(f"Output already exists; enable resume or choose another path: {self.output_path}")
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
            validate_payload(payload)
            previous_source = Path(payload["source_pointcloud"]).expanduser().resolve()
            if previous_source != self.pointcloud_path:
                raise SystemExit(
                    "Existing annotation belongs to another point cloud:\n"
                    f"  existing: {previous_source}\n  current:  {self.pointcloud_path}"
                )
            previous_ground_z = float(payload.get("ground_plane", {}).get("z_m", self.ground_z_m))
            previous_camera_height = float(
                payload.get("camera", {}).get("height_above_ground_m", self.camera_height_m)
            )
            if not np.isclose(previous_ground_z, self.ground_z_m):
                raise SystemExit(
                    f"Existing annotation uses ground_z_m={previous_ground_z}, current config uses {self.ground_z_m}."
                )
            if not np.isclose(previous_camera_height, self.camera_height_m):
                raise SystemExit(
                    "Existing annotation uses camera_height_m="
                    f"{previous_camera_height}, current config uses {self.camera_height_m}."
                )
            self.payload = payload
            print(f"\n继续已有标注：{self.output_path}")
            print(f"  已有目标数量：{len(self.targets)}")
            return
        self.payload = new_annotation_payload(
            source_pointcloud=self.pointcloud_path,
            ground_z_m=self.ground_z_m,
            camera_height_m=self.camera_height_m,
            max_display_points=self.max_display_points,
            loaded_display_points=len(self.display_pcd.points),
            display_center=self.display_center,
            defaults=self.defaults,
        )

    def next_default_id(self) -> str:
        used = {item["target_id"] for item in self.targets}
        index = len(self.targets) + 1
        while f"target_{index:03d}" in used:
            index += 1
        return f"target_{index:03d}"

    def next_equipment_id(self) -> str:
        used = {str(item.get("equipment_id")) for item in self.targets if item.get("equipment_id")}
        index = len(used) + 1
        while f"equipment_{index:03d}" in used:
            index += 1
        return f"equipment_{index:03d}"

    def prompt_equipment(self) -> dict[str, str]:
        equipment_id = self.next_equipment_id()
        equipment_type = input(f"设备类型 [{self.defaults.category}]: ").strip() or self.defaults.category
        while True:
            equipment_name = input("设备名称（同一物理设备的唯一名称）: ").strip()
            if equipment_name:
                break
            print("设备名称不能为空，例如：1号主变、断路器101。")
        print(f"设备编号：{equipment_id}")
        print(f"本批次随后选择的所有三维点都归属于：{equipment_name} ({equipment_type})")
        return {
            "equipment_id": equipment_id,
            "equipment_name": equipment_name,
            "equipment_type": equipment_type,
        }

    def next_inspection_point_id(self, equipment_id: str) -> str:
        count = sum(1 for item in self.targets if item.get("equipment_id") == equipment_id)
        return f"{equipment_id}_point_{count + 1:03d}"

    def prompt_target_record(self, target_xyz: np.ndarray, equipment: dict[str, str]) -> dict[str, Any]:
        used_ids = {item["target_id"] for item in self.targets}
        default_id = self.next_default_id()
        target_id = default_id
        if self.prompt_target_id:
            while True:
                target_id = input(f"目标ID [{default_id}]: ").strip() or default_id
                if target_id in used_ids:
                    print(f"目标ID已存在：{target_id}，请重新输入。")
                    continue
                break
        else:
            print(f"自动目标ID：{target_id}")
        inspection_point_id = self.next_inspection_point_id(equipment["equipment_id"])
        default_point_name = f"inspection_point_{inspection_point_id.rsplit('_', 1)[-1]}"
        inspection_point_name = input(f"巡视点位名称 [{default_point_name}]: ").strip() or default_point_name
        label = f"{equipment['equipment_name']}/{inspection_point_name}"
        task_type = self.defaults.task_type
        if self.prompt_task_type:
            task_type = input(f"巡视任务类型 [{self.defaults.task_type}]: ").strip() or self.defaults.task_type
        min_distance = self.defaults.min_observation_distance_m
        max_distance = self.defaults.max_observation_distance_m
        exclusion_radius = self.defaults.target_exclusion_radius_m
        if self.prompt_observation_parameters:
            while True:
                min_distance = prompt_float("最小观测距离/m", self.defaults.min_observation_distance_m)
                max_distance = prompt_float("最大观测距离/m", self.defaults.max_observation_distance_m)
                exclusion_radius = prompt_float("目标端射线排除半径/m", self.defaults.target_exclusion_radius_m)
                try:
                    validate_observation_parameters(min_distance, max_distance, exclusion_radius)
                except ValueError as exc:
                    print(f"观测参数无效：{exc}")
                    print("请重新输入三个参数。")
                    continue
                break
        else:
            print(
                "使用默认观测参数："
                f"距离 {min_distance:g}～{max_distance:g} m，"
                f"目标端排除半径 {exclusion_radius:g} m"
            )
        return make_target_record(
            target_id=target_id,
            label=label,
            category=equipment["equipment_type"],
            task_type=task_type,
            target_xyz=target_xyz,
            ground_z_m=self.ground_z_m,
            min_observation_distance_m=min_distance,
            max_observation_distance_m=max_distance,
            target_exclusion_radius_m=exclusion_radius,
            notes="",
            annotation_id=len(self.targets) + 1,
            source_pointcloud=self.pointcloud_path,
            camera_height_m=self.camera_height_m,
            equipment_id=equipment["equipment_id"],
            equipment_name=equipment["equipment_name"],
            equipment_type=equipment["equipment_type"],
            inspection_point_id=inspection_point_id,
            inspection_point_name=inspection_point_name,
        )

    def save_picked_target(self, picked_display: np.ndarray, equipment: dict[str, str]) -> dict[str, Any]:
        target_xyz = np.asarray(picked_display, dtype=np.float64) + self.display_center
        print(f"  选中显示坐标：{picked_display}")
        print(f"  恢复后的完整点云坐标：{target_xyz}")
        print(f"  相对基准地面高度：{target_xyz[2] - self.ground_z_m:.3f} m")
        record = self.prompt_target_record(target_xyz, equipment)
        self.targets.append(record)
        save_payload(self.output_path, self.payload)
        print(
            f"已保存 {record['target_id']}："
            f"{record['equipment_name']}/{record['inspection_point_name']}；"
            f"当前共 {len(self.targets)} 个巡视点位。"
        )
        print(f"输出文件：{self.output_path}")
        return record

    def show_review_window(self) -> None:
        if not self.targets:
            return
        print("\n[三维目标标注 3/3] 打开目标复核窗口")
        print("  红色球体表示已保存的三维巡视目标；按Q关闭窗口。")
        vis = self.o3d.visualization.Visualizer()
        vis.create_window(window_name="三维巡视目标复核", width=1280, height=800)
        vis.add_geometry(self.display_pcd)
        for target in self.targets:
            sphere = self.o3d.geometry.TriangleMesh.create_sphere(radius=self.review_sphere_radius_m)
            sphere.paint_uniform_color([1.0, 0.05, 0.02])
            sphere.translate(np.asarray(target["target_xyz"], dtype=np.float64) - self.display_center)
            vis.add_geometry(sphere)
        vis.add_geometry(coordinate_frame_for_points(self.o3d, np.asarray(self.display_pcd.points), ratio=0.05))
        configure_visualizer(vis, point_size=self.point_size)
        configure_default_camera(vis)
        vis.run()
        vis.destroy_window()

    def run(self) -> None:
        print("\n" + "=" * 72)
        print("三维点云巡视目标标注")
        print("=" * 72)
        print("标注对象是设备表面的三维巡视目标，不是机器人二维停靠点。")
        print("安全停靠区域将在后续由观测距离、三维遮挡和二维安全地图自动计算。")
        self.load_pointcloud()
        self.prepare_payload()
        print("\n[三维目标标注 2/3] 按设备选择巡视点位")
        print("  使用SceneWidget + Open3DScene，可单独控制黄色标记点大小。")
        print("  请选择真实需要观察的设备表面，不要选择设备内部或几何中心。")
        equipment = self.prompt_equipment()
        picker = SceneWidgetInspectionPicker(
            self.o3d,
            self.display_pcd,
            point_size=self.point_size,
            marker_point_size=self.selection_marker_point_size,
            background_color=self.display_background_color,
        )
        quit_all = False
        try:
            while True:
                picker.start_device()
                print("请一次完成该设备的全部巡视点位，然后再切换到下一台设备。")
                while True:
                    action, picked_display = picker.wait_for_action(equipment["equipment_name"])
                    if action == "confirm_point" and picked_display is not None:
                        picker.run_terminal_task(lambda: self.save_picked_target(picked_display, equipment))
                        picker.commit_selected_point(picked_display)
                        continue
                    if action == "quit":
                        quit_all = True
                    break
                print(f"设备“{equipment['equipment_name']}”的巡视点位标注结束。")
                continue_next = False
                if not quit_all:
                    continue_next = bool(
                        picker.run_terminal_task(
                            lambda: ask_yes_no("是否继续标注下一台设备？", default=False)
                        )
                    )
                if quit_all or not continue_next:
                    break
                equipment = picker.run_terminal_task(self.prompt_equipment)
        finally:
            picker.close()
        if self.show_review:
            self.show_review_window()
        if self.targets:
            save_payload(self.output_path, self.payload)
        print("\n三维巡视目标标注结束。")
        print(f"  目标数量：{len(self.targets)}")
        print(f"  输出文件：{self.output_path}")
