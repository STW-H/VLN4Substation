# Algorithm Overview

本文档记录当前项目的算法主线和数据处理流程。环境安装、外部依赖和运行命令放在根目录 `README.md` 中说明。

当前二维语义代价 A* 的公式、参数和具体实施步骤见 `docs/two_dimensional_astar_technical_plan.md`。

## 1. 主要思路

当前项目采用“离线传统规划负责安全到达可行巡视区域，在线 VLN 负责局部主动观测”的分阶段架构。全局规划不预测机器人到达后的真实图像质量，也不依赖 VLN/VLM 产生底层安全路径。

在变电站巡检任务中，机器人并不需要依赖 VLN/VLM 完成完整路径规划。全局路径、可通行区域、安全距离和禁入区域约束，更适合由传统地图和规划算法完成：

```text
完整点云 / 二维地图 / 语义地图
离线全局规划
在线局部规划与避障
安全距离和禁入区域检查
```

VLN 的主要价值放在候选停靠区域内的局部主动观测，而不是替代传统导航：

```text
判断巡视目标是否进入视野
判断当前观测角度是否满足拍摄需求
根据现场遮挡、设备外观和任务语义调整观测位姿
提高 RGB / 热成像采集质量
减少固定机位在遮挡、角度不佳时带来的漏检风险
```

因此，当前算法主线可以概括为：

```text
1. 完整点云建立统一的三维坐标系，并将拟合基准地面设置为 z=0
2. 二维语义地图经过障碍物膨胀形成统一硬安全空间
3. 在三维离散点云中，根据观测距离和视线遮挡计算地面可见候选区域
4. 将可见候选区域与二维硬安全空间求交，得到可行巡视区域
5. 区域目标 A* 联合选择安全停靠位置和全局路径
6. 道路、狭窄空间、障碍净空和优势路径作为软代价，不再构造硬分层规划空间
7. 机器人到达候选区域后，由局部 VLN 根据真实图像调整最终观测位姿
8. VLM 解析巡视指令；环境风险由天气先验、现场传感器和机器人视觉共同确定
```

论文方法的核心不是单独改进 A* 的搜索形式，而是建立“三维目标可见性—二维运动安全—上下文语义代价”的耦合规划框架。硬安全约束始终不随任务变化；任务紧急程度和环境风险只调节软代价权重。

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

随后交互选择若干相互分散的平坦地面区域拟合基准地平面，以平面法向确定 Z+；再选择两个结构点确定 X+，Y+ 由正交叉乘得到。坐标轴旋转后只沿新 Z 轴平移，使拟合基准地面的质心位于 `z=0`，X/Y 的处理方式保持不变。地面点受扫描噪声、沟槽和实际坡度影响，会分布在零点附近，不要求所有地面点严格等于零。

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
  annotate.py                           # 统一的二维语义/三维巡视目标标注入口
  merge_annotation_files.py             # 多次标注结果合并

visualization/
  view_pointcloud.py                    # 查看完整点云和普通点云
  view_gaussian.py                      # 使用 Habitat-GS 查看/渲染高斯

planning/
  build_planning_map.py                 # 从合并标注构建规划地图
  build_feasible_inspection_regions.py  # 从三维目标生成二维可行巡视区域
  run_baseline_astar.py                 # 二维 A* 基线规划与验证
```

站点专用、一次性、可由官方工具替代的脚本不再放在 `tools/` 中。

`src/substation_vln/` 中保存可复用实现代码，并按功能域组织：

```text
annotation/       # 标注数据结构与交互标注器
preprocessing/    # 点云/高斯/坐标预处理基础函数
visualization/    # 点云与高斯可视化基础函数
planning/         # 路径规划算法，含 baseline A* 与 improved A*
inspection_regions/ # 三维可见性检测与可行巡视区域生成
```

配置文件统一放在：

```text
substation_vln/configs/
```

其中，当前二妃山规划地图构建使用：

```text
substation_vln/configs/tools/planning/build_planning_map_erfeishan.yaml
```

每个 `tools` 脚本也都有对应的默认 YAML 配置，位于 `substation_vln/configs/tools/` 下。新工具的配置与脚本使用完全相同的基名，例如 `build_feasible_inspection_regions.py` 对应 `build_feasible_inspection_regions.yaml`。命令行入口默认读取自己的 YAML，同时仍允许通过命令行参数临时覆盖配置。

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
狭窄空间区域
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

为了生成可行巡视区域，需要先在轴矫正后的完整点云中定义巡视目标。当前将每个巡视目标简化为一个三维表面点，而不是预先固定唯一机器人停靠点。

需要标注：

```text
设备名称和类别
物理设备唯一名称 equipment_name
设备分组ID equipment_id
设备内巡视点位名称 inspection_point_name
巡视目标三维坐标 target_xyz
最小观测距离 d_min
最大观测距离 d_max
目标端射线排除半径
RGB / 热成像任务类型
```

三维目标与二维语义标注使用统一的时间戳文件：

```text
outputs/<site>/annotation/annotation_<timestamp>.json
```

三维记录使用 `category=inspection_target` 和 `geometry_type=point_3d`，保存人工选取的目标点及观测参数，不保存由算法推导出的可见性掩膜或巡视区域。标注时必须一次完成同一物理设备的全部巡视点位；这些点共享 `equipment_id` 和 `equipment_name`，每个点具有独立的 `inspection_point_id` 和 `inspection_point_name`，以支持后续多目标覆盖和最少停靠点优化。它与表示人工二维停靠位姿的 `patrol_point` 含义不同。目标点必须位于需要观察的设备表面，例如表盘中心或状态指示区域，不能位于设备内部。机器人摄像头暂时假设为固定高度的全向相机，因此离线区域计算不考虑相机 yaw、视场角和目标表面法向；真实图像是否合格及最终位姿调整由在线局部 VLN 负责。

派生数据在二维标注合并并构建规划地图后统一计算，流程为：

```text
annotate.py --mode target3d             # 人工标注三维巡视目标
merge_annotation_files.py              # 合并二维语义与三维目标标注
build_planning_map.py                   # 构建稳定的二维硬安全空间
build_feasible_inspection_regions.py    # 批量生成各目标的可行巡视区域
```

这样，修改二维障碍物、膨胀参数或三维可见性参数时，只需重新运行派生计算，不需要重新标注三维目标。

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
  -> narrow_space_mask
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
narrow_space 作为可通行但具有额外软风险代价的狭窄空间
preferred_path 作为有向或无向路径先验的中心线
patrol_point 作为后续巡视任务点和巡视方向
```

`narrow_space` 支持多边形、矩形和圆形标注。规划地图将其栅格化为 `narrow_space_mask`，保持在 `free_space_mask` 内可通行，并通过配置参数增加软风险代价。

派生类地图由基础地图和配置参数生成，主要服务于 A* 和改进 A*：

```text
obstacle_inflation_radius_m 控制障碍物膨胀半径
preferred_path_sigma_m 控制路径吸引场扩散范围
preferred_path_alpha 控制路径吸引强度
narrow_space_penalty 控制狭窄空间附加代价
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
├── narrow_space_mask.png
├── preferred_path_mask.png
├── cost_map.png
└── planning_overlay.png
```

当前交互式 A* 工程测试主要读取：

```text
free_space_mask
cost_map
patrol_points
grid metadata
```

该测试用于验证坐标转换、可达性、结果保存和可视化链路。论文实验中的传统 baseline A* 不应直接使用包含语义先验的 `cost_map`，否则 baseline 会提前包含 preferred road、preferred path 和障碍物软代价，导致对比不公平。

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

## 5. 论文方法设计

### 5.1 研究定位

近年的改进 A* 已广泛采用双向搜索、动态启发函数、冗余节点删除、转向代价和平滑等方法。单独组合这些常见改进，或人工设置“效率、标准、保守”三组权重，难以形成充分的论文贡献。

当前论文拟研究：

```text
基于三维点云可行巡视区域和上下文语义风险的变电站巡检区域目标规划
```

核心问题分为两个相互关联的部分：

```text
去哪里停：由三维点云中的目标距离和视线遮挡生成地面可见候选区域，再与二维硬安全空间求交得到可行巡视区域
如何到达：在统一硬安全空间内，根据道路、狭窄空间、障碍净空和任务环境上下文构造软代价
```

最终由区域目标 A* 在可行巡视区域中自动选择路径代价较低的终点。机器人到达该区域后的真实观测质量评价和短距离调整属于在线 VLN 阶段，不进入离线全局规划目标函数。

### 5.2 统一硬安全空间与语义软代价

当前假设人工标注障碍物经过机器人安全半径膨胀后，已经包含现阶段需要考虑的硬安全约束：

```text
S_safe = planning_boundary - inflated_obstacle
```

项目不再构造道路层、普通层和完整层等互斥或嵌套的硬规划空间。道路和狭窄空间只改变路径代价：

```text
planning_boundary / inflated_obstacle  决定能不能走
preferred_road                          表示优先通行
narrow_space                            表示可以通行但应承担额外风险代价
preferred_path                          表示位置或方向先验
distance_to_obstacle                    表示软净空代价
```

如果某个区域在任何正常条件下都不允许进入，应标为障碍物或禁入区，而不是只使用狭窄空间软惩罚。

路径 `P` 的语义代价可写为：

```text
J(P | z) = wL(z) * J_length(P)
         + wR(z) * J_road(P)
         + wN(z) * J_narrow(P)
         + wC(z) * J_clearance(P)
         + wD(z) * J_direction(P)
         + wT(z) * J_turn(P)
```

其中 `z` 表示任务和环境上下文。硬安全空间不随 `z` 改变，只有软代价权重发生变化。

### 5.3 基于离散点云的可行巡视区域生成

#### 5.3.1 输入与假设

离线可行巡视区域计算只使用轴矫正后的完整点云，不依赖 3D Gaussian。当前作如下简化：

```text
拟合基准地面为 z=0
机器人停靠位置在地面的投影为 g=(x,y)
相机安装高度为 h_camera
相机三维位置为 c(g)=(x,y,h_camera)
巡视目标为设备表面三维点 t_i=(x_i,y_i,z_i)
相机暂时视为全向，不考虑 yaw 和视场角
离线阶段只判断观测距离与点云视线遮挡
```

若实际场地存在不可忽略的坡道或地面高程变化，后续再将固定 `z=0` 扩展为局部地面高度图；当前二妃山站点先采用平面地面假设。

#### 5.3.2 点云离散化

原始点云是非均匀离散采样，不能简单以“射线上是否恰好存在点”作为遮挡条件。应将完整点云体素化为三维占据栅格：

```text
V(i,j,k) = 1  体素被占据
V(i,j,k) = 0  体素未被占据
```

体素分辨率和最小占据点数需要根据点云密度确定。必要时对占据体素做一格左右的小范围膨胀，减少射线从稀疏点云孔洞穿过造成的假可见；体素过大或膨胀过强也会产生假遮挡，因此需要参数敏感性实验。

#### 5.3.3 距离筛选

对二维地面候选位置 `g=(x,y)`，计算相机到目标的三维距离：

```text
d_i(g) = ||c(g) - t_i||_2
```

距离候选区域为：

```text
G_distance_i = {g | d_min_i <= d_i(g) <= d_max_i}
```

这一步先排除过近和过远位置，只有剩余候选点需要执行三维射线检测。

#### 5.3.4 点云遮挡判断

从相机位置 `c(g)` 向目标点 `t_i` 构造三维线段：

```text
r(s) = c(g) + s * (t_i - c(g)),  s in [0,1]
```

使用 3D DDA 或 Amanatides-Woo 算法遍历线段经过的占据体素。如果射线中段遇到占据体素，则该候选点被遮挡。

射线检测必须忽略两端：相机端排除一小段机器人附近空间；目标端使用 `target_exclusion_radius_m` 排除目标自身点云。目标点应位于待观察设备表面，不能标在设备内部，否则设备自身会被误判为遮挡。

可见区域定义为：

```text
G_visible_i = {g in G_distance_i | LOS(c(g), t_i; V) = 1}
```

### 5.4 三维可见性与二维安全地图耦合

候选位置直接在二维规划栅格上生成。对于每个栅格中心，利用 `grid_to_xy` 得到工程坐标 `(x,y)`，构造相机位置并执行三维距离与遮挡检查，因此输出天然与二维规划地图逐格对应：

```text
visible_inspection_region_mask_i[row, col] = distance_valid and line_of_sight_valid
```

再与二维硬安全空间求交：

```text
feasible_inspection_region_mask_i = visible_inspection_region_mask_i & free_space_mask
```

三维点云负责判断“从这里能否在合适距离内看到目标”，二维地图负责判断“机器人能否安全停在这里”。两类约束保持解耦，不需要在完整三维空间规划机器人运动轨迹。

### 5.5 区域目标 A*

传统 A* 的终点是固定栅格；本文将 `feasible_inspection_region_mask_i` 作为目标区域：

```text
终止条件：current cell in feasible_inspection_region_mask_i
启发函数：当前栅格到目标区域的最小二维距离
```

目标区域距离可预先使用二维距离变换生成。区域目标规划联合选择停靠位置与路径：

```text
(P*, g*) = argmin J(P | z)
subject to:
  P lies in S_safe
  endpoint(P) = g*
  g* in G_feasible_inspection_i
```

离线阶段不再预测目标在真实图像中的清晰度、居中程度或检测置信度。只要终点满足距离、无遮挡和二维安全要求，即视为合格候选；到达后的实际观测效果由局部 VLN 处理。

### 5.6 任务与现场环境驱动的代价自适应

效率、标准、保守三种模式可以作为固定权重基线或典型工作点，但不作为论文方法主体。论文方法研究连续上下文到代价权重的映射：

```text
w = f(z)
z = [任务紧急程度, 环境风险, 定位可靠度, 续航压力, ...]
```

VLM可以从自然语言巡视指令中提取：

```text
目标设备
巡视部件
任务类型
时间要求
文本中的紧急语义
```

最终紧急程度不能只由 VLM 自由决定，还应融合站内告警等级、人工指定等级和设备任务规则，并由确定性规则模块校验。

天气预报只作为环境先验。实际环境状态应融合：

```text
天气接口：降雨/降雪概率、温度、风速和能见度
站内传感器：温湿度、降雨、路面温度等定量信息
机器人现场视觉/VLM：积水、积雪、疑似结冰、低能见度和临时占用
```

湿度等不可可靠从 RGB 图像直接测量的变量必须来自传感器。积水、积雪和临时障碍等局部现象还应形成空间风险图，而不是全部压缩为一个全局模式。所有自适应结果只改变软代价，不能放松 `S_safe`。

### 5.7 全局规划与局部 VLN 的职责边界

```text
离线全局阶段：
  点云体素化
  目标距离筛选
  三维遮挡检测
  二维安全区域求交
  上下文语义代价构建
  区域目标 A*

在线局部阶段：
  获取真实图像
  判断目标是否进入视野及观测是否合格
  在安全动作集合内由 VLN 调整位置
  满足停止条件后完成图像采集
```

局部 VLN 的在线成败不反向加入当前离线规划约束。后续可以单独研究失败恢复或全局重规划，但不属于第一版离线可行巡视区域算法。

### 5.8 Baseline A*

论文中的 baseline A* 应保持为传统栅格 A*：

```text
输入：free_space_mask、起点、终点
状态：(row, col)
邻域：4 邻域或 8 邻域
实际代价：直线步长 1，对角步长 sqrt(2)
启发函数：Manhattan、Euclidean 或 Octile distance
终点：人工固定巡视点或固定安全停靠点
不使用：三维可见区域、preferred road、narrow space、preferred path 和障碍物软排斥
```

这样才能分别量化三维可见性约束、可行巡视区域、区域目标搜索、语义代价和上下文自适应的收益。

## 6. 对比与消融实验

### 6.1 算法组

建议至少设置：

```text
B0  固定人工巡视点 + 传统 A*
B1  目标 XY 最近安全点 + 传统 A*
B2  仅使用观测距离环 + 区域目标 A*
B3  距离环 + 二维投影视线判断 + 区域目标 A*
M1  三维点云可见区域 + 二维安全约束 + 区域目标 A*
M2  M1 + 固定语义代价
M3  M2 + 任务环境上下文自适应代价（完整方法）
```

效率、标准和保守三组人工固定参数可作为附加基线，与连续自适应权重比较。可选外部搜索基线包括 Dijkstra、JPS 和 Theta*；局部 VLN 不与静态全局规划的搜索时间混为同一指标。

### 6.2 测试任务

```text
使用多个三维巡视目标点和多个二维可行起点构造任务
增加规划边界内随机可行起点
覆盖固定点被遮挡、多个分离可见区域、窄通道和道路绕行场景
重点构造二维投影无遮挡、但三维高度上被设备遮挡的案例
构造短路径经过狭窄空间、长路径经过宽阔道路的风险—效率冲突案例
建议形成不少于 100 组有效规划任务
```

### 6.3 评价指标

```text
路径长度
规划时间、扩展节点数和峰值内存
可观测候选区域面积与连通分量数量
终点三维可见率、假可见率和假遮挡率
固定巡视点失效时的规划成功率
最小/平均障碍物距离
preferred road 内路径比例
狭窄空间内路径长度
到 preferred path 的平均距离
有向路径方向遵循率
转弯次数和累计转角
规划成功率
```

同时报告均值、标准差和显著性检验，不应只展示少数视觉效果较好的路径。

### 6.4 参数敏感性

需要重点分析：

```text
地图分辨率：0.05 m、0.10 m
三维体素分辨率
体素最小占据点数和占据膨胀半径
相机高度 h_camera
最小/最大观测距离
目标端射线排除半径
障碍物膨胀半径：根据机器人半宽和安全余量设置
preferred_path_sigma_m
preferred_path_alpha
preferred_road_cost
狭窄空间、障碍净空和方向代价权重
紧急程度与环境风险到代价权重的映射参数
```

地图参数和算法参数应分开保存。每次实验结果需要记录实际配置文件、地图 metadata、起终点和随机种子。

## 7. 相关文献

1. Yin et al., “An Improved A-Star Path Planning Algorithm Based on Mobile Robots in Medical Testing Laboratories,” Sensors, 2024. [DOI: 10.3390/s24061784](https://doi.org/10.3390/s24061784)
2. “Improved A* Path Planning Method Based on the Grid Map,” Sensors, 2022. [DOI: 10.3390/s22166198](https://doi.org/10.3390/s22166198)
3. “Combined improved A* and greedy algorithm for path planning of multi-objective mobile robot,” Scientific Reports, 2022. [DOI: 10.1038/s41598-022-17684-0](https://doi.org/10.1038/s41598-022-17684-0)
4. “Mobile Robot Path Planning Based on Kinematically Constrained A-Star Algorithm and DWA Fusion Algorithm,” Mathematics, 2023. [DOI: 10.3390/math11214552](https://doi.org/10.3390/math11214552)
5. “Research on global path planning algorithm for mobile robots based on improved A*,” Expert Systems with Applications, 2023. [DOI: 10.1016/j.eswa.2023.122922](https://doi.org/10.1016/j.eswa.2023.122922)
6. “Path Planning for Substation Inspection Robots Based on a Fusion Algorithm Incorporation JPS and DWA,” IEEE Access, 2024. [DOI: 10.1109/ACCESS.2024.3478769](https://doi.org/10.1109/ACCESS.2024.3478769)
7. “Improved A* and DWA fusion algorithm based path planning for intelligent substation inspection robot,” Measurement and Control, 2025. [DOI: 10.1177/00202940251316687](https://doi.org/10.1177/00202940251316687)
8. “Path Planning Trends for Autonomous Mobile Robot Navigation: A Review,” Sensors, 2025. [DOI: 10.3390/s25041206](https://doi.org/10.3390/s25041206)

## 8. 后续工作

后续主要内容将围绕以下方向继续完善：

```text
拆分纯传统 baseline A* 与当前语义 cost 工程测试
重新生成地面 z=0 的正射图、语义标注和规划地图
将 narrow_space 标注转换为栅格 mask 和软风险代价
定义三维巡视目标点及其观测距离参数
构建完整点云三维体素占据地图
实现相机到目标点的三维体素射线遮挡检测
生成 visible_inspection_region_mask 和 feasible_inspection_region_mask
实现区域目标 A* 及目标区域距离启发图
实现固定语义代价与上下文自适应代价的对比方法
生成 preferred path 切向量场
完成消融实验和参数敏感性分析
实现局部 VLN 主动观测和安全动作接口
实现 VLM 指令解析及现场环境多源融合接口
仿真验证流程
```

当前已经完成点云坐标预处理、地面归零、Gaussian 配准工具链、语义标注、规划地图构建和交互式 A* 链路验证。下一阶段首先在新坐标系中重新生成二维标注和规划地图，然后实现三维点云可见区域与区域目标 A*。语义风险自适应和局部 VLN 在该离线主链路验证完成后接入。
