# 三维巡视目标标注说明

## 1. 标注对象

本工具在轴矫正后的完整点云中标注设备表面的三维巡视目标：

```text
三维巡视目标：需要被相机观察的设备表面点
二维巡视停靠点：机器人预先指定的固定停车位置
安全停靠区域：后续根据距离、遮挡和二维安全地图自动生成的区域
```

三者不能混用。新的三维标注不直接指定机器人在哪里停车。

目标点应选择在真实待观察表面，例如表盘中心、开关状态指示区域或油位显示区域。不要选择设备内部或设备几何中心，否则后续点云射线可能先击中设备表面，将目标误判为被遮挡。

## 2. 软件结构

核心实现：

```text
substation_vln/src/substation_vln/annotation/inspection_targets_3d.py
```

二维地图与三维目标共用命令行入口：

```text
substation_vln/tools/annotation/annotate.py
```

统一配置：

```text
substation_vln/configs/tools/annotation/annotate.yaml
```

配置与脚本使用完全相同的基名 `annotate`。入口通过模式菜单或 `--mode map/target3d` 调用相应底层标注器，二维与三维交互实现仍在 `src` 中解耦。

核心模块负责点云加载与显示中心化、交互点选、参数校验、目标记录、续标兼容性、自动保存和最终复核；`tools`脚本只负责解析YAML和命令行参数。

## 3. 运行方式

在项目根目录执行：

```bash
conda run -n habitat-gs python \
  substation_vln/tools/annotation/annotate.py --mode target3d
```

默认输入：

```text
substation_vln/data/processed/220kv_erfeishan/pointcloud/
erfeishan_0.02_resampled_real_coords_axis_corrected.ply
```

默认输出与二维标注采用相同的时间戳命名：

```text
substation_vln/outputs/220kv_erfeishan/annotation/annotation_<timestamp>.json
```

## 4. Open3D操作

标注以物理设备为批次。开始一个批次时先输入设备类型，再输入设备名称，然后连续点选该设备的全部巡视点位：

```text
输入设备类型
→ 输入设备名称
→ 自动分配 equipment_id
→ 在SceneWidget窗口中Shift+左键选择一个三维巡视点
→ Shift+右键可取消候选点，Enter确认候选点
→ 窗口隐藏并在命令行输入该巡视点位名称
→ 保存后窗口自动恢复，并以黄色点显示已记录位置
→ 按Q完成当前设备
→ 选择是否标注下一台设备
```

每次点选一个目标：

```text
Shift + 鼠标左键：选择候选点
Shift + 鼠标右键：取消当前尚未确认的候选点
Enter：确认候选点并进入命令行输入点位名称
Q：完成当前设备
```

每次 `Shift+左键` 只设置一个黄色候选点。若位置不合适，可用 `Shift+右键` 取消；按 Enter 确认后窗口隐藏并显示：

```text
临时中心化后的显示坐标
恢复后的完整工程XYZ坐标
目标相对z=0基准地面的高度
```

输入巡视点位名称后立即保存，然后自动恢复同一设备的点云窗口。黄色标记显示当前设备已经保存的全部点位，大小由 `selection_marker_point_size` 控制。Q只负责完成当前设备；若按Q时还有未按Enter确认的候选点，该候选点会被放弃。程序随后询问标注下一台设备或结束。

程序为设备自动生成 `equipment_001`、`equipment_002`，并为巡视点生成 `target_001` 和 `equipment_001_point_001` 等ID。设备名称用于识别同一物理设备，设备类型只需每台设备输入一次；每个三维点只需输入巡视点位名称。任务类型、观测距离和目标端射线排除半径默认读取YAML。确认后立即写入JSON，标注阶段不计算可行巡视区域。

点选窗口使用 `SceneWidget + Open3DScene`，以支持独立黄色标记、逐点命名、撤销和Q结束设备。退出连续标注后，程序按原流程打开一次最终复核窗口。

完成三维目标标注、二维标注合并和规划地图构建后，使用独立脚本生成可行巡视区域：

```text
substation_vln/tools/planning/build_feasible_inspection_regions.py
```

## 5. 参数说明

参数位于：

```text
substation_vln/configs/tools/annotation/annotate.yaml
```

| 参数 | 含义 | 当前默认值 |
|---|---|---:|
| `pointcloud` | 轴矫正后的完整点云 | 二妃山处理点云 |
| `output` | 时间戳标注JSON；空值时自动生成 | `null` |
| `max_display_points` | Open3D最大显示/可选点数 | `20000000` |
| `point_size` | 点云显示点尺寸 | `3.0` |
| `selection_marker_point_size` | 点选窗口中黄色已记录点位尺寸 | `8.0` |
| `review_sphere_radius_m` | 最终复核窗口的目标标记球半径 | `0.3 m` |
| `ground_z_m` | 基准地面高度 | `0.0 m` |
| `camera_height_m` | 后续全向相机离地高度 | `1.0 m` |
| `default_equipment_type` | 默认设备类型 | `unknown_device` |
| `default_task_type` | 默认巡视类型 | `visual_inspection` |
| `default_min_distance_m` | 默认最小观测距离 | `5.0 m` |
| `default_max_distance_m` | 默认最大观测距离 | `20.0 m` |
| `default_exclusion_radius_m` | 目标自身点云排除半径 | `0.1 m` |
| `prompt_target_id` | 是否逐目标询问ID | `false` |
| `prompt_task_type` | 是否逐目标询问巡视类型 | `false` |
| `prompt_observation_parameters` | 是否逐目标询问观测参数 | `false` |
| `no_resume` | 禁止向已有文件追加 | `false` |
| `no_review` | 跳过最终三维复核 | `false` |

点云显示会按照固定间隔采样。采样只影响用户能够点选的显示点，不改变保存坐标系。保存前会把临时中心化偏移加回，`target_xyz`始终处于完整轴矫正点云坐标系。

## 6. 输出格式

输出JSON采用通用标注文件结构，顶层字段包括：

```text
schema_version
annotation_kind
saved_at
source_pointcloud
coordinate_frame
ground_plane
camera
display_sampling
default_observation_parameters
categories
annotations
```

单个目标示例：

```json
{
  "target_id": "breaker_001",
  "equipment_id": "equipment_001",
  "equipment_name": "1号断路器",
  "equipment_type": "circuit_breaker",
  "inspection_point_id": "equipment_001_point_001",
  "inspection_point_name": "状态指示面板",
  "label": "1号断路器/状态指示面板",
  "category": "inspection_target",
  "device_category": "circuit_breaker",
  "selection_type": "pointcloud_point",
  "geometry_type": "point_3d",
  "task_type": "state_inspection",
  "target_xyz": [519100.0, 3344600.0, 2.4],
  "target_height_above_ground_m": 2.4,
  "min_observation_distance_m": 2.0,
  "max_observation_distance_m": 6.0,
  "target_exclusion_radius_m": 0.2,
  "notes": ""
}
```

观测参数必须满足：

```text
0 <= min_observation_distance_m < max_observation_distance_m
0 <= target_exclusion_radius_m < max_observation_distance_m
```

目标、设备和设备内巡视点位的ID均自动编号。合并多个文件时，`merge_annotation_files.py` 会统一重新编号，并保留 `source_target_id`、`source_equipment_id` 和 `source_inspection_point_id`。设备名称应能唯一指向物理设备；同名设备会被视为同一设备并合并巡视点位。

## 7. 继续标注与数据安全

默认 `no_resume: false`。如果输出文件已存在，程序会验证：

```text
JSON类型正确
目标ID没有重复
源点云路径一致
ground_z_m一致
camera_height_m一致
```

验证通过后在原文件上继续追加。若需要禁止覆盖或追加，可运行：

```bash
python substation_vln/tools/annotation/annotate.py --mode target3d --no-resume
```

每个确认目标都会立即保存，不必等到整次程序结束。

## 8. 与停靠区域计算的接口

三维目标文件与二维标注文件先统一合并。后续停靠区域计算器读取：

```text
annotations_merged.json 中 category=inspection_target、geometry_type=point_3d 的记录
轴矫正后的完整点云
planning_map.npz
planning_map_metadata.json
```

对二维地面候选点 `g=(x,y)` 构造相机位置：

```text
c(g) = (x, y, ground_z_m + camera_height_m)
```

再根据每个目标的距离范围和目标端排除半径执行三维体素视线通道检测，最终与 `free_space_mask` 求交生成 `feasible_inspection_region_mask`。
