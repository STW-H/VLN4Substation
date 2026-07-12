# Algorithm Overview

本文档记录当前项目的算法主线和数据处理流程。环境安装、外部依赖和运行命令放在根目录 `README.md` 中说明。

## 1. 主要思路

当前项目采用“传统导航负责安全可达，视觉语言模型负责局部主动观测”的分层架构。

在变电站巡检任务中，机器人并不需要依赖 VLN/VLM 完成完整路径规划。全局路径、可通行区域、安全距离和禁入区域约束，更适合由传统地图和规划算法完成：

```text
完整点云 / 二维地图 / 语义地图
离线全局规划
在线局部规划与避障
安全距离和禁入区域检查
```

VLN/VLM 的主要价值放在候选巡视点附近的局部主动观测，而不是替代传统导航：

```text
判断巡视目标是否进入视野
判断当前观测角度是否满足拍摄需求
根据现场遮挡、设备外观和任务语义调整观测位姿
提高 RGB / 热成像采集质量
减少固定机位在遮挡、角度不佳时带来的漏检风险
```

因此，当前算法主线可以概括为：

```text
1. 传统地图与规划模块将机器人送至候选巡视区域
2. 安全约束模块持续限制机器人可运动空间
3. 高斯场景提供视觉仿真和观测效果检查
4. VLN/VLM 在局部范围内优化最终观测位姿
5. 机器人完成巡检图像采集和结果记录
```

## 2. 完整点云与高斯点云预处理流程

对于一个新的变电站，如果同时拥有完整点云和 3D Gaussian 数据，当前建议按以下流程完成预处理。

### 2.1 数据整理

原始数据建议放在：

```text
substation_vln/data/raw/<site_name>/
├── pointcloud/       # LAS / LAZ / LiData / PLY 等完整点云
├── gaussian/         # 原始 3D Gaussian PLY
├── metadata/         # GPS、采集说明、坐标说明等
└── viewer/           # 原始查看器或辅助文件，可选
```

处理后的数据建议放在：

```text
substation_vln/data/processed/<site_name>/
├── pointcloud/
├── gaussian/
├── registration/
├── maps/
├── navmesh/
├── semantic/
└── safety/
```

### 2.2 完整点云预处理

完整点云作为主要参考坐标系。若输入是 LAS/LAZ，需要先转换为真实坐标 PLY：

```text
LAS 整数坐标
  -> 使用 LAS header 中的 scale / offset
  -> 转换为真实坐标
  -> 保存为 binary PLY
```

当前工具：

```text
substation_vln/tools/preprocessing/convert_las_to_real_ply.py
substation_vln/tools/visualization/view_pointcloud.py --save-converted
```

转换后需要检查：

```text
点数是否正确
XYZ 范围是否合理
RGB 是否正常保留
坐标是否处于真实工程坐标系
```

### 2.3 高斯点云预处理

原始 3D Gaussian 文件首先保留在 `raw/gaussian/` 中。当前二妃山数据中，原始高斯为 Z-up；为了避免混淆，项目不把 Y-up 副本作为常规预处理结果保存到 `processed/`。

处理流程：

```text
原始 Gaussian PLY
  -> 保留在 raw/gaussian/ 作为基准高斯数据
  -> 使用 Habitat-GS 查看时，由 view_gaussian.py 自动生成或复用 outputs/ 下的 Y-up 可视化缓存
  -> 在配准工具中以高斯中心点形式检查结构与颜色
  -> 后续若生成滤波、对齐、坐标矫正后的高斯点云，再保存到 processed/gaussian/
```

当前工具：

```text
substation_vln/tools/visualization/view_gaussian.py
```

说明：Habitat-GS 内部仍使用 Y-up 坐标习惯。`view_gaussian.py` 会优先查找 `outputs/220kv_erfeishan/gaussian_yup_cache/` 中的缓存文件；若不存在或原始高斯更新，则重新生成。该缓存属于可视化输出，可以删除并自动重建；正式数据管理仍以 Z-up 原始高斯为基准。

### 2.4 高斯到完整点云配准

当前以坐标轴矫正后的完整点云作为参考坐标系，将原始 Z-up 高斯中心点注册到完整点云。

配准流程：

```text
1. 在高斯点云中选取若干清晰结构点
2. 在完整点云中按相同顺序选择对应结构点
3. 使用 Umeyama 相似变换完成粗配准
4. 对参与 ICP 的高斯点云进行范围、高度和统计离群点过滤
5. 对完整点云和高斯点云执行 ICP 精配准
6. 保存 Gaussian -> PointCloud 的最终变换矩阵
7. 保存对齐后的高斯点云到 `processed/gaussian/`
8. 在同一窗口中可视化完整点云和变换后的高斯点云，人工确认效果
```

当前工具：

```text
substation_vln/tools/preprocessing/register_gaussian_to_pointcloud.py
```

当前基准变换保存位置：

```text
substation_vln/data/processed/220kv_erfeishan/registration/gaussian_to_pointcloud_transform.json
```

该 JSON 应包含：

```text
手工对应点
粗配准矩阵
ICP 参数和结果
最终 Gaussian -> PointCloud 变换矩阵
```

### 2.5 预处理结果

完成预处理后，应得到以下基础数据：

```text
完整点云真实坐标 PLY
完整点云坐标轴矫正 PLY
raw/gaussian/ 中的原始 Z-up 高斯文件
Gaussian -> PointCloud 坐标变换矩阵
processed/gaussian/ 中对齐后的高斯点云
点云和高斯的可视化检查结果
```

这些数据为后续地图生成、语义标注、安全约束构建和局部观测位姿优化提供统一坐标基础。

### 2.6 当前保留的工具脚本

当前 `substation_vln/tools/` 按流程模块组织通用命令行入口：

```text
preprocessing/
  convert_las_to_real_ply.py            # LAS/LAZ 转真实坐标 PLY
  register_gaussian_to_pointcloud.py    # Gaussian 到完整点云配准
  render_pointcloud_ortho_image.py      # 从点云生成标注用正射图

annotation/
  annotate_ortho_image.py               # 正射图标注
  merge_annotation_files.py             # 多次标注结果合并

visualization/
  view_pointcloud.py                    # 查看完整点云和普通点云
  view_gaussian.py                      # 使用 Habitat-GS 查看/渲染高斯

planning/
  build_planning_map.py                 # 从合并标注构建规划地图
```

站点专用、一次性、可由官方工具替代的脚本不再放在 `tools/` 中。

`src/substation_vln/` 中保存可复用实现代码，并按功能域组织：

```text
annotation/       # 标注数据结构与交互标注器
preprocessing/    # 点云/高斯/坐标预处理基础函数
visualization/    # 点云与高斯可视化基础函数
planning/         # 路径规划算法，含 baseline A* 与 improved A*
```

配置文件统一放在：

```text
substation_vln/configs/
```

其中，当前二妃山规划地图构建使用：

```text
substation_vln/configs/tools/planning/build_planning_map_erfeishan.yaml
```

每个 `tools` 脚本也都有对应的默认 YAML 配置，位于 `substation_vln/configs/tools/` 下。命令行入口默认读取自己的 YAML，同时仍允许通过命令行参数临时覆盖配置。

## 3. 预处理后的标注流程

完成坐标对齐后，下一阶段进入标注流程。当前阶段只定义标注目标和基本流程，具体标注工具和数据格式后续再完善。

### 3.1 可通行区域标注

首先需要沿完整点云 Z 方向生成 X-Y 平面的高清俯视底图，然后在该底图上标注机器人可通行区域。俯视图像需要同时保存像素坐标和真实世界坐标之间的转换关系，便于后续将标注 mask 还原到地图坐标系。

底图生成结果应包括：

```text
Z 方向俯视 RGB 图像
图像分辨率，单位 m/pixel
图像宽高
点云 XYZ 范围
pixel -> world XY 坐标转换关系
world XY -> pixel 坐标转换关系
```

需要标注：

```text
道路区域
可行驶硬化地面
不可通行设备区
围栏、墙体、沟槽等障碍
临时或永久禁入区域
```

目标产物：

```text
二维占据地图
可通行区域 mask
不可通行区域 mask
地图分辨率和坐标原点说明
```

### 3.2 安全约束标注

变电站机器人运动必须满足安全距离要求，因此需要在地图上标注安全约束。

需要标注：

```text
带电设备安全距离
设备外轮廓或安全缓冲区
道路边界
禁止靠近区域
机器人最大允许活动范围
```

目标产物：

```text
safety mask
inflation / buffer 参数
禁入区域多边形
安全距离配置文件
```

### 3.3 巡检目标与候选观测点标注

为了支持后续局部主动观测，需要标注巡检对象和候选观测区域。

需要标注：

```text
设备名称和类别
巡检目标位置
推荐初始观测点
允许局部调整的搜索范围
推荐拍摄方向或目标朝向
RGB / 热成像采集要求
```

目标产物：

```text
inspection targets
candidate viewpoints
target-viewpoint association
task instruction templates
```

### 3.4 语义地图标注

为了让后续语言任务和地图导航结合，需要建立基础语义层。

可选语义包括：

```text
道路
设备区
主变 / 开关 / 刀闸 / 支架 / 绝缘子等设备类别
围栏
门架
道路交叉口
巡视点
危险区
```

目标产物：

```text
semantic map
semantic object list
object id 与空间范围
语义名称与任务语言的映射关系
```

## 4. 规划地图构建

完成标注合并后，下一步是把矢量标注转换为路径规划可直接使用的栅格地图。当前实现分为两层：

```text
基础类地图构建
  -> boundary_mask
  -> obstacle_mask
  -> preferred_road_mask
  -> preferred_path_mask
  -> patrol_points

派生类地图构建
  -> inflated_obstacle_mask
  -> free_space_mask
  -> distance_to_obstacle_m
  -> distance_to_preferred_path_m
  -> preferred_path_attraction
  -> cost_map
```

基础类地图只表达人工标注直接给出的空间语义：

```text
planning_boundary 作为规划活动边界
obstacle 作为不可通行障碍物
preferred_road 作为优先通行区域
preferred_path 作为有向路径吸引子的中心线
patrol_point 作为后续巡视任务点和巡视方向
```

派生类地图由基础地图和配置参数生成，主要服务于 A* 和改进 A*：

```text
obstacle_inflation_radius_m 控制障碍物膨胀半径
preferred_path_sigma_m 控制路径吸引场扩散范围
preferred_path_alpha 控制路径吸引强度
preferred_road_cost 控制优先道路代价
obstacle_repulsion_radius_m / obstacle_repulsion_weight 控制贴近障碍物的惩罚
resolution_m 控制栅格地图分辨率
```

当前工具：

```text
substation_vln/tools/planning/build_planning_map.py
substation_vln/tools/planning/run_baseline_astar.py
```

运行命令：

```bash
python substation_vln/tools/planning/build_planning_map.py \
  --config substation_vln/configs/tools/planning/build_planning_map_erfeishan.yaml
```

默认输出：

```text
substation_vln/outputs/220kv_erfeishan/planning/maps/
├── planning_map.npz
├── planning_map_metadata.json
├── patrol_points.json
├── boundary_mask.png
├── obstacle_mask.png
├── inflated_obstacle_mask.png
├── free_space_mask.png
├── preferred_road_mask.png
├── preferred_path_mask.png
├── cost_map.png
└── planning_overlay.png
```

后续实现 baseline A* 时，应主要读取：

```text
free_space_mask
cost_map
patrol_points
grid metadata
```

改进 A* 可以进一步利用：

```text
preferred_path_attraction
distance_to_preferred_path_m
distance_to_obstacle_m
preferred_road_mask
```

baseline A* 的第一版交互测试流程为：

```text
1. 打开规划地图窗口
2. 鼠标左键点击起始点
3. 在命令行选择目标巡视点编号
4. A* 在 free_space_mask 内搜索路径
5. 保存路径 JSON 和路径叠加图
6. 显示规划结果
```

当前 baseline A* 配置：

```text
substation_vln/configs/tools/planning/run_baseline_astar_erfeishan.yaml
```

默认输出：

```text
substation_vln/outputs/220kv_erfeishan/planning/baseline_astar/
```

## 5. 后续工作

后续主要内容将围绕以下方向继续完善：

```text
候选巡视点和观测位姿数据结构
baseline A* 路径规划
引入 preferred_path 吸引场和障碍物距离场的改进 A*
局部主动观测策略
VLM/VLN 黑盒评价与局部决策接口
仿真验证流程
```

当前阶段先以“完成可靠预处理、坐标统一、标注合并和规划地图构建”为主，确保后续 A*、改进 A* 和局部观测算法都建立在同一坐标基础上。
