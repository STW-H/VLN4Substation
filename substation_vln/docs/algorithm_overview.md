# 变电站 VLN 巡检算法总览

## 1. 工程背景

变电站机器狗巡检同时包含两类性质不同的任务：

- 设备之间的长距离运动要求安全、稳定、可解释；
- 到达设备附近后，需要根据自然语言指定的部件灵活调整底盘与云台并完成拍摄。

完全依赖固定巡视点，会产生大量现场定点工作，设备或道路变化后还要重新维护；完全依赖 VLN 进行整站导航，又难以保证对围栏、沟槽、带电设备和道路边界等工程约束的严格遵守。因此，本项目采用分层方法：

```text
操作票/自然语言
    ↓
远程大模型解析为结构化拍摄任务
    ↓
改进式 A* 完成设备间全局安全运动和目标停靠位姿选择
    ↓
局部 VLN 在设备邻域内搜索指定部件并触发拍摄
```

VLN 只负责局部导航与拍摄，不判断设备正常、异常或故障。状态识别由人工或独立视觉分析模块完成。

## 2. 解题思路

### 2.1 离线地图承担确定性安全约束

完整点云经过地面校正后生成正射俯视图。人工只标注规划必需的信息：

- 规划边界；
- 障碍物；
- 优先通过区；
- 优先路径；
- 狭窄空间；
- 巡视设备占地区域和设备名称、类型。

标注保存在 `data/annotations/<site>/`。一次绘制多个设备圆时，每个圆保存为独立设备记录并自动编号，避免多个设备被当作一个整体计算。

`build_planning_map.py` 将标注栅格化，生成两种 A* 共用的边界、障碍物、道路、优先路径、设备索引和语义代价层。障碍物是硬约束，靠近障碍物是软惩罚，道路与优先路径提供引导代价。

### 2.2 终点由固定点改为可行目标位姿区域

巡视设备不是一个必须精确到达的点。对每台设备，系统从标注范围内的点云估计三维包围盒，在指定观察高度区间内取观察中心，并根据相机俯仰角范围反算允许距离：

\[
d(\theta)=\frac{z_{roi}-z_{cam}}{\tan\theta}.
\]

俯仰角上下限形成二维环形候选区。候选位置还必须同时满足：

- 机器狗 `0.8 m × 0.4 m` 矩形足迹不与边界、障碍物或设备碰撞；
- 机身方向近似与设备中心圆相切，便于沿道路继续运动；
- 云台水平和俯仰角处于硬件范围；
- 相机到观察中心的点云视线通道没有其他设备或结构遮挡。

推荐俯仰角是软目标。偏离推荐角度不会直接删除候选位姿，而是形成终点代价，使 A* 在行驶距离和拍摄姿态之间联合选择。

### 2.3 改进式 A* 联合选择路径和停靠位姿

改进式 A* 的状态为：

\[
s=(r,c,k),
\]

其中 `(r,c)` 是栅格位置，`k` 是离散机身航向。搜索目标不是单一终点，而是一组包含位置、机身方向和云台姿态的可行终端状态。

总代价由四部分组成：

```text
路径长度
+ 道路、优先路径、狭窄空间和障碍物距离代价
+ 机身旋转与横向运动代价
+ 终点相机俯仰角代价
```

为了控制三维状态空间的计算量，当前采用两阶段搜索：

1. 二维区域目标 A* 先寻找从起点到任意可行目标位置的粗路径；
2. 将粗路径扩张为走廊，在走廊内执行带机身航向的位姿 A*；若失败，逐步扩大走廊。

最终输出完整路径、终点机器狗位姿、云台 `pan/tilt` 和各项代价。详细公式和实现对应关系见 [improved_astar_technical_design.md](improved_astar_technical_design.md)。

### 2.4 局部 VLN 解决固定机位难以处理的问题

改进式 A* 只负责把机器人安全送到能够看到目标设备的初始区域。进入设备邻域后，局部 VLN 根据设备类型、指定部件、现场图像以及设备资料完成：

- 判断当前位于设备的哪个方向；
- 控制前进、后退、转向和云台左右/上下观察；
- 绕开临时障碍；
- 找到指定部件并触发拍摄；
- 返回拍摄图像及机器人、云台位姿。

这样可以减少逐部件固定巡视点标注，同时保留全局运动的确定性安全边界。

## 3. 当前软件实现

### 3.1 主要模块

```text
tools/preprocessing/
  convert_las_to_real_ply.py
  register_gaussian_to_pointcloud.py
  render_pointcloud_ortho_image.py

tools/annotation/
  annotate.py
  merge_annotation_files.py

tools/tasks/
  parse_inspection_instruction.py

tools/planning/
  build_planning_map.py
  run_baseline_astar.py
  build_inspection_goal_regions.py
  run_region_goal_astar.py

src/substation_vln/planning/
  astar/                 # 固定终点普通 A* baseline
  common/                # 公共栅格、语义代价和输入输出
  improved_astar/        # 设备几何、目标位姿、遮挡检测和区域目标 A*
```

普通 A* 保留为对照算法，不参与改进式目标区域生成。

### 3.2 数据流

```text
完整点云
  ├─> 正射图 ─> 人工标注 ─> annotations_merged.json
  └──────────────────────────────┐
                                 ↓
annotations_merged.json ─> planning_map.npz
                                 ↓
点云 + 设备索引 ─> inspection_goal_regions.npz
                                 ↓
起点 + 目标设备 ─> region-goal pose A* ─> 路径与终点位姿
```

所有可再生地图、缓存、规划结果和复核图保存到 `outputs/<site>/`；原始数据、处理后基础数据和人工标注保存在 `data/`。

### 3.3 当前设备观察配置

- 变压器：观察高度 `0%–100%`，俯仰角 `30°–60°`，推荐 `45°`；
- 其他设备：观察高度 `50%–100%`，俯仰角 `30°–70°`，推荐 `45°`；
- 相机高度：`1.0 m`；
- 机器狗尺寸：长 `0.8 m`、宽 `0.4 m`；
- 机身航向：16 个离散方向；
- 视线遮挡：基于 `0.2 m` 点云体素的加粗中心视线通道。

这些值均由 `build_inspection_goal_regions.yaml` 配置，不写死在算法中。

## 4. 标准运行顺序

```bash
python substation_vln/tools/annotation/annotate.py
python substation_vln/tools/annotation/merge_annotation_files.py
python substation_vln/tools/planning/build_planning_map.py
python substation_vln/tools/planning/build_inspection_goal_regions.py
python substation_vln/tools/planning/run_region_goal_astar.py
```

设备标注或设备编号改变后，需要重新合并标注、构建规划地图，并使用 `--rebuild-geometry` 重建设备点云几何。只有点云本身改变时才需要 `--rebuild-visibility`。

## 5. 当前边界与后续工作

当前已经完成二维地图、设备目标位姿区域、点云中心视线遮挡和分层区域目标位姿 A*。局部 VLN 控制器尚未实现，任务解析模块目前通过 DeepSeek API 远程调用。

现有遮挡检测只检查观察区域中心附近的加粗视线通道，不能保证整个设备轮廓完全无遮挡。后续可扩展为多视线可见率，但不应将其与当前已验证的中心视线基线混为一套实验结果。
