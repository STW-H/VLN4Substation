# Algorithm Overview

本文档记录当前项目的算法主线和数据处理流程。环境安装、外部依赖和运行命令放在根目录 `README.md` 中说明。

当前二维语义代价 A* 的公式、参数和具体实施步骤见 `docs/two_dimensional_astar_technical_plan.md`。

## 1. 主要思路

当前项目采用“离线传统规划负责设备间长距离安全导航，在线 VLN 负责设备邻域内的局部搜索与拍摄”的分层架构。全局规划不依赖大模型产生底层安全路径，VLN也不负责判断设备正常、异常或故障。

在变电站巡检任务中，机器人并不需要依赖 VLN/VLM 完成完整路径规划。全局路径、可通行区域、安全距离和禁入区域约束，更适合由传统地图和规划算法完成：

```text
完整点云 / 二维地图 / 语义地图
离线全局规划
在线局部规划与避障
安全距离和禁入区域检查
```

VLN 的主要价值放在设备目标位姿邻域内的局部拍摄，而不是替代传统全局导航：

```text
根据自然语言指定的设备和部件执行局部视觉搜索
根据现场遮挡、设备外观和任务语义调整机器人与相机姿态
触发拍摄并保存图像、机器人位姿和云台角度
适应临时障碍物和设备型号变化，减少固定机位维护工作量
将采集结果交给人工或独立分析模块，不输出设备状态结论
```

因此，当前算法主线可以概括为：

```text
1. 完整点云建立统一坐标系，并生成用于二维人工标注的正射底图
2. 人工标注规划边界、道路、障碍物、狭窄空间和设备占地区域
3. DeepSeek远程文本API将操作票或自然语言解析为有序的结构化拍摄任务
4. 通过设备名称将拍摄任务关联到二维语义地图中的目标设备
5. 从设备区域内点云估计设备三维轮廓，并利用相机成像约束生成可行目标位姿区域
6. 分层区域目标位姿A*联合选择长距离路径、机器狗停车位置/方向和云台初始姿态
7. 到达目标位姿区域后，局部VLN控制底盘和云台搜索指定部件并触发拍摄
8. 拍摄事件完成后记录任务状态并切换到下一设备；故障判断不属于本文范围
```

论文方法的核心不是单独改进A*或直接使用大模型控制整站运动，而是建立“自然语言任务解析—二维设备语义定位—区域目标全局规划—局部视觉语言拍摄”的分层巡视框架。二维地图和安全控制提供确定性约束，VLN提供设备邻域内的语义灵活性。

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
  annotate.py                           # 二维地图语义与巡视设备区域标注
  merge_annotation_files.py             # 多次标注结果合并

tasks/
  parse_inspection_instruction.py       # 远程DeepSeek解析操作票/自然语言拍摄任务

visualization/
  view_pointcloud.py                    # 查看完整点云和普通点云
  view_gaussian.py                      # 使用 Habitat-GS 查看/渲染高斯

planning/
  build_planning_map.py                 # 从合并标注构建规划地图
  run_baseline_astar.py                 # 二维 A* 基线规划与验证
  build_inspection_goal_regions.py      # 设备几何与相机可行目标位姿预计算
  run_region_goal_astar.py              # 分层区域目标位姿 A*
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

完成坐标对齐后进入二维标注流程。当前工具已经支持地图语义和巡视设备占地区域，并统一保存像素坐标与工程XY坐标。

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

### 3.3 巡视设备区域标注

当前主流程不再逐点标注三维巡视目标，也不人工绘制固定可停靠范围。离线人工标注只描述设备名称、类型和二维占地区域；A*与局部VLN之间的交接位姿由规划模块计算。具体部件搜索、遮挡适应、视角调整和拍摄由局部VLN完成。

标注数据使用统一的时间戳文件：

```text
data/annotations/<site>/sessions/annotation_<timestamp>.json  # 矢量标注主数据
data/annotations/<site>/sessions/annotation_<timestamp>.png   # 同名2K复核缩略图
```

每次标注必须同时生成JSON和最长边2048像素的同名PNG；缩略图只用于快速复核，规划始终读取JSON。多次会话合并后对应生成 `annotations_merged.json` 和 `annotations_merged.png`。

设备标注只保存稳定的语义和几何信息：

```text
equipment_region
  人工标注设备占地轮廓、设备名称和类型
  设备名称应与操作票、设备台账和自然语言指令中的唯一名称一致
```

接近距离、机器狗停车位置/方向以及云台姿态均不属于人工标注。规划模块以设备指定高度段的中心为圆锥顶点，根据俯仰范围生成相机高度平面上的圆环，再结合矩形足迹、静态点云遮挡、自由空间和语义代价生成初始交接位姿。全局规划只要求机器人能看到该ROI中心，以便局部VLN判断当前相对方位。

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
固定巡视点（仅用于baseline）
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
  -> equipment_mask
  -> equipment_index_mask
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
patrol_point 仅作为固定终点A* baseline的巡视点和方向
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

当前规划链路已经增加逐设备的区域数据：

```text
equipment_mask / equipment_index_mask  设备占地栅格及设备索引
feasible_goal_poses_i                 相机与矩形足迹约束后的逐设备目标位姿集合
```

`build_planning_map.py` 输出两种A*共用的设备、道路、障碍物和代价层；`build_inspection_goal_regions.py` 在这些通用层上生成改进A*专用目标位姿。`patrol_points.json` 仅继续服务于固定终点A*基线。

当前工具：

```text
substation_vln/tools/planning/build_planning_map.py
substation_vln/tools/planning/run_baseline_astar.py
substation_vln/tools/planning/build_inspection_goal_regions.py
substation_vln/tools/planning/run_region_goal_astar.py
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
├── equipment_regions.json
├── boundary_mask.png
├── obstacle_mask.png
├── inflated_obstacle_mask.png
├── free_space_mask.png
├── preferred_road_mask.png
├── narrow_space_mask.png
├── preferred_path_mask.png
├── equipment_mask.png
├── cost_map.png
├── pose_cost_map.png
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
pose_center_space_mask
pose_cost_map
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

改进A*共享上述规划地图，但另外读取：

```text
equipment_regions.json       设备名称、类型和工程坐标轮廓
equipment_index_mask         每个栅格所属的设备编号
boundary/obstacle/equipment  矩形足迹碰撞检查的硬约束
preferred_road/path/narrow   路径搜索的语义软代价
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

近年的改进A*已经广泛采用双向搜索、动态启发函数、冗余节点删除、转向代价和平滑等方法。单独组合这些常见改进，或简单把A*与通用大模型串联，均难以形成充分的论文贡献。

当前论文拟研究：

```text
面向自然语言拍摄任务的变电站全局区域规划—局部VLN分层巡视方法
```

该方法针对固定巡视点工程存在的四类问题：

```text
逐点定姿与后期维护工作量大
更换设备或部件位置变化后需要重新定点
固定机位难以适应临时遮挡和现场环境变化
操作票、自然语言和设备资料难以直接参与传统定点程序
```

本文不要求离线地图确定唯一精确拍摄点。区域目标A*负责把机器人安全送到目标设备附近，局部VLN再根据现场图像、自然语言部件名称和设备知识完成短距离搜索、云台调整与拍摄。拍摄结果的故障分析由人工或独立算法处理。

论文贡献应集中在以下三点：

```text
1. 以相机可行目标位姿区域代替唯一固定巡视点的全局—局部交接机制
2. 操作票/自然语言到地图设备及局部拍摄任务的结构化语义连接
3. 融合设备多角度资料的局部VLN拍摄策略及跨设备适应性
```

### 5.2 统一硬安全空间与语义软代价

baseline A*使用圆形近似膨胀障碍物，其硬安全空间为：

```text
S_safe = planning_boundary - inflated_obstacle
```

改进A*不再以圆形膨胀近似机器狗，而是对每个离散航向使用旋转矩形足迹腐蚀原始自由空间：

```text
S_raw = planning_boundary - obstacle - equipment
S_safe(k) = Erode(S_raw, rotated_robot_footprint(k))
```

`pose_cost_map`在 `S_raw` 内保持有限值，不把baseline的圆形膨胀带作为第二个硬约束。当前矩形不增加额外安全余量；与障碍物的净空偏好改由距离软代价表达，默认排斥半径为1.5 m、权重为2.0。无论软代价多大，真实旋转矩形与障碍物相交的状态始终不可通行。

项目不再构造道路层、普通层和完整层等互斥或嵌套的硬规划空间。道路和狭窄空间只改变路径代价：

```text
baseline: planning_boundary / inflated_obstacle 决定能不能走
improved: planning_boundary / obstacle / equipment / rotated footprint 决定能不能走
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

### 5.3 自然语言巡视任务解析与地图语义绑定

操作票或自然语言指令首先通过远程DeepSeek文本API解析为确定格式的拍摄任务。当前电脑不部署本地大语言模型，API密钥只从环境变量 `DEEPSEEK_API_KEY` 读取，不写入配置或数据文件。

结构化任务定义为：

```text
T_k = (sequence, equipment_name, equipment_type,
       inspection_part, action, image_count, requested_views)
```

其中 `action` 当前只能是 `capture`。例如：

```json
{
  "task_id": "task_001",
  "sequence": 1,
  "equipment_name": "1号主变",
  "equipment_type": "main_transformer",
  "inspection_part": "油位计",
  "action": "capture",
  "image_count": 1,
  "requested_views": []
}
```

解析器不得虚构原指令未给出的设备编号、部件、视角或拍摄数量。解析结果还必须使用设备名称和设备台账校验，并与 `equipment_region.equipment_name` 匹配。匹配失败时应请求人工确认，不能让语言模型自行猜测地图目标。

当前实现：

```text
substation_vln/tools/tasks/parse_inspection_instruction.py
substation_vln/src/substation_vln/tasks/instruction_parser.py
substation_vln/src/substation_vln/tasks/schema.py
```

### 5.4 相机可行目标位姿区域生成

离线阶段根据设备二维占地区域和指定三维高度段生成机器狗终端位姿集合。这里不直接设置停车距离；内外半径由ROI中心高度、相机高度和设备俯仰范围解析反算。

```text
设备点云 = 设备二维轮廓内点云 - 地面点 - 离群点
设备中心 = 分位数过滤后设备点云包围盒中心
候选状态 = (x, y, robot_yaw, camera_pan, camera_tilt)
```

机器狗在二维地图中建模为长0.8 m、宽0.4 m的有向矩形。当前不额外扩大矩形安全余量；机身长轴与以设备中心为圆心的圆相切，因此每个候选位置有两个相反的切向航向。旋转矩形必须完整位于规划边界内，且不得与障碍物或设备本体相交。

相机水平航向指向ROI中轴，俯仰角与相机到ROI中心的锥线仰角一致。算法不检查设备水平轮廓和上下边缘是否完整入镜；其目的不是在全局停靠点完成最终拍摄，而是让局部VLN看到设备并判断当前观察方位。满足矩形碰撞、云台角度和ROI中心视线约束的状态组成目标位姿区域。

当前实现：

```text
substation_vln/src/substation_vln/planning/improved_astar/equipment_geometry.py
substation_vln/src/substation_vln/planning/improved_astar/camera_model.py
substation_vln/src/substation_vln/planning/improved_astar/goal_pose_region.py
substation_vln/src/substation_vln/planning/improved_astar/visibility.py
substation_vln/tools/planning/build_inspection_goal_regions.py
```

#### 5.4.1 输入、输出与执行位置

目标位姿区域属于可重新生成的规划中间结果，不属于人工标注。输入为：

```text
annotations_merged.json                 人工标注的设备名称、类型和二维轮廓
planning_map.npz                        通用栅格地图及设备索引
planning_map_metadata.json              GridSpec和地图参数
axis_corrected_pointcloud.ply           地面归零后的完整工程坐标点云
build_inspection_goal_regions.yaml      机器人、相机和采样参数
```

输出默认位于：

```text
outputs/<site>/planning/inspection_goal_regions/
├── equipment_geometry.json
├── inspection_goal_regions.npz
├── inspection_goal_regions_metadata.json
└── inspection_goal_regions_overlay.png
```

其中 `equipment_geometry.json` 是设备三维几何缓存，`inspection_goal_regions.npz` 保存候选终端位姿及压缩后的逐航向碰撞可行掩码，PNG只用于人工复核。

#### 5.4.2 大规模点云中的设备几何提取

当前完整点云约有6755万个点，不能为每台设备重复完整读取。实现采用二进制PLY内存映射并分块单次扫描。对点 \(p_j=(x_j,y_j,z_j)\)，先按规划栅格映射：

\[
c_j=\left\lfloor\frac{x_j-x_{\min}}{\rho}\right\rfloor,
\qquad
r_j=\left\lfloor\frac{y_{\max}-y_j}{\rho}\right\rfloor .
\]

若 `equipment_index_mask[r_j,c_j]=i`，则该点属于第 \(i\) 个设备的二维柱状裁剪范围。随后删除：

```text
不在规划栅格范围内的点
非有限坐标点
z <= ground_z_m + ground_clearance_m 的地面点
超出设定分位数范围的离群点
```

当前默认地面高度为0 m，地面过滤余量为0.15 m，稳健范围使用1%和99.5%分位数。记过滤后的设备点集为 \(Q_i\)，其稳健坐标上下界为：

\[
q_i^{\min}=\operatorname{Percentile}_{1\%}(Q_i),
\qquad
q_i^{\max}=\operatorname{Percentile}_{99.5\%}(Q_i).
\]

稳健包围盒中心用于设备二维定位；其高度上下界再按设备配置生成ROI中心：

\[
c_i^{3D}=\frac{q_i^{\min}+q_i^{\max}}{2}.
\]

同时保留点云坐标中位数及稳健上下界用于诊断。当前停车区域计算使用二维设备中轴位置和指定高度段中心，不再为每个候选投影完整三维包络盒。

配置中的 `scan_stride` 只影响三维几何提取采样，不改变二维碰撞地图。设备名称或二维轮廓变化时缓存会自动失效；如果底层点云更新但名称和轮廓不变，应显式使用 `--rebuild-geometry`。

#### 5.4.3 按设备类型选择实际观测本体

完整点云范围用于描述设备几何。设设备稳健高度为 \([z_{min},z_{max}]\)，配置垂直观测比例为 \([f_{min},f_{max}]\)，ROI高度范围为：

\[
z_{obs,min}=z_{min}+f_{min}(z_{max}-z_{min}),
\]

\[
z_{obs,max}=z_{min}+f_{max}(z_{max}-z_{min}).
\]

圆锥顶点高度为该线段中心：

\[
z_{aim}=\frac{z_{obs,min}+z_{obs,max}}{2}.
\]

当前配置为：

```text
zhubian                         0.0--1.0，观察完整设备
duanluqi / daozha / bileiqi    0.5--1.0，只观察上半部功能本体
default                         0.5--1.0
```

主变使用0%--100%高度，圆锥顶点位于完整设备高度中心；其他设备使用50%--100%高度，顶点位于上半部中心。后续若任务指定具体巡视部位，可直接用该部位包络盒中心替换圆锥顶点。

#### 5.4.4 有向矩形机器狗足迹

机器狗中心状态为 \((x,y,\theta)\)，本体尺寸为：

\[
L=0.8\ \mathrm m,\qquad W=0.4\ \mathrm m.
\]

安全余量 \(m_s\) 同时加到四周，因此碰撞检查尺寸为：

\[
L_s=L+2m_s,\qquad W_s=W+2m_s.
\]

当前默认 `safety_margin_m=0.0`，即碰撞包络就是0.8 m × 0.4 m机器狗本体。基础可行空间使用：

\[
S_{raw}=M_{boundary}\cap\neg M_{obstacle}\cap\neg M_{equipment}.
\]

对每个离散航向 \(\theta_k\)，将旋转后的矩形栅格化为结构元素 \(K_k\)，然后腐蚀基础可行空间：

\[
S_k=S_{raw}\ominus K_k.
\]

若 `pose_free_masks[k,r,c]=1`，表示以该栅格为中心、航向为 \(\theta_k\) 的完整矩形足迹满足硬碰撞约束。这里不直接复用baseline的圆形障碍膨胀，而是按航向检查真实矩形包络。

#### 5.4.5 与设备中心圆相切的机身方向

设设备二维中心为 \(c_i=(x_i,y_i)\)，候选停车中心为 \(p=(x_p,y_p)\)，径向角为：

\[
\phi(p)=\operatorname{atan2}(y_p-y_i,x_p-x_i).
\]

机器狗长轴与中心圆相切时有两个相反候选方向：

\[
\theta_t^{+}=\phi+\frac{\pi}{2},
\qquad
\theta_t^{-}=\phi-\frac{\pi}{2}.
\]

航向按 `heading_bins` 离散。默认16个方向，对应22.5°分辨率。两个方向分别执行矩形碰撞和相机约束检查，A*最终根据到达路径、转向和视觉代价选择其中之一。

#### 5.4.6 云台姿态与针孔投影

相机位置由机器狗中心、机身航向、安装高度及前后/左右安装偏移计算。设相机世界坐标为 \(t_c\)，设备二维中轴位置为 \((x_i,y_i)\)，相机水平航向固定指向设备中轴：

\[
\psi_c=\operatorname{atan2}(y_i-y_c,x_i-x_c),
\]

云台相对机身的水平角为：

\[
\psi_{pan}=\operatorname{wrap}(\psi_c-\theta_{robot}).
\]

当前设备类型分别配置俯仰范围和推荐角度：

```text
bianyaqi: 观察高度0%--100%，俯仰20°--60°，推荐30°
default:  观察高度50%--100%，俯仰30°--70°，推荐45°
```

这里的角度定义为相对水平面向上为正。相机俯仰直接等于相机位置到ROI中心的连线仰角，不再使用图像边缘或相机视场角。

#### 5.4.7 ROI圆锥与距离圆环

设ROI中心为 \(q=(x_q,y_q,z_q)\)，相机高度为 \(h_c\)，高度差为：

\[
\Delta h=z_q-h_c.
\]

对俯仰角 \(\vartheta\)，圆锥与相机高度平面的交圆半径为：

\[
 r(\vartheta)=\frac{\Delta h}{\tan\vartheta}.
\]

角度区间 \([\vartheta_{min},\vartheta_{max}]\) 对应圆环：

\[
r_{min}=\frac{\Delta h}{\tan\vartheta_{max}},\qquad
r_{max}=\frac{\Delta h}{\tan\vartheta_{min}}.
\]

候选相机位置必须落在该圆环内。其实际俯仰角由位置反算，推荐角度使用设备配置中的 \(\vartheta_{pref}\)，分段归一化软代价为：

\[
J_{tilt}=\begin{cases}
\dfrac{\vartheta_{pref}-\vartheta}{\vartheta_{pref}-\vartheta_{min}},&\vartheta\le\vartheta_{pref},\\
\dfrac{\vartheta-\vartheta_{pref}}{\vartheta_{max}-\vartheta_{pref}},&\vartheta>\vartheta_{pref}.
\end{cases}
\]

推荐角度处代价为0，配置范围两端代价为1。解析圆环随后与规划边界、逐航向机器狗足迹和ROI中心无遮挡视线相交，得到最终可行目标位姿区域。

#### 5.4.8 候选采样与保存格式

为了避免在0.05 m栅格上保存大量几乎相同的终点，默认每4个栅格采样一次，即候选位置间隔0.2 m；矩形碰撞仍使用0.05 m掩码。对每个采样位置计算相机光心到ROI中轴的水平距离并反算唯一俯仰角。只有距离和俯仰均落入圆锥壳区间的候选才继续执行两个切向机身方向的矩形碰撞与点云视线检查。

`inspection_goal_regions.npz` 主要字段为：

```text
pose_free_packed            按位压缩的[heading,row,col]足迹可行掩码
goal_equipment_index        候选位姿所属设备编号
goal_rows / goal_cols       候选停车栅格
goal_heading_bins           候选机身航向编号
goal_tilt_costs             归一化俯仰角代价
goal_camera_pan_rad         云台水平角
goal_camera_tilt_rad        云台俯仰角
```

#### 5.4.9 目标位姿生成参数位置

配置文件为：

```text
substation_vln/configs/tools/planning/build_inspection_goal_regions.yaml
```

主要参数与代码含义如下：

| 配置项 | 当前默认值 | 实现含义 |
|---|---:|---|
| `robot.length_m` | 0.8 | 机器狗矩形本体长度 |
| `robot.width_m` | 0.4 | 机器狗矩形本体宽度 |
| `robot.safety_margin_m` | 0.0 | 矩形四周额外安全余量，当前关闭 |
| `robot.heading_bins` | 16 | 机身航向离散数量 |
| `camera.image_width_px` | 1920 | 投影图像宽度 |
| `camera.image_height_px` | 1080 | 投影图像高度 |
| `camera.height_m` | 1.0 | 相机光心离地高度 |
| `camera.forward_offset_m` | 0.0 | 相机相对机身中心的前向偏移 |
| `camera.lateral_offset_m` | 0.0 | 相机相对机身中心的左向偏移 |
| `camera.pan_min/max_deg` | -180 / 180 | 云台水平限位 |
| `camera.tilt_min/max_deg` | 20 / 70 | 覆盖所有设备配置的云台全局硬限位 |
| `camera.preferred_tilt_deg` | 45 | 未配置设备类型时的推荐俯仰角回退值 |
| `observation_profiles.default.tilt_min/max_deg` | 30 / 70 | 非主变设备俯仰范围 |
| `observation_profiles.default.preferred_tilt_deg` | 45 | 非主变设备推荐俯仰角 |
| `observation_profiles.bianyaqi.tilt_min/max_deg` | 20 / 60 | 主变俯仰范围 |
| `observation_profiles.bianyaqi.preferred_tilt_deg` | 30 | 主变推荐俯仰角 |
| `observation_profiles.<type>.vertical_min_fraction` | 依设备类型 | 从设备稳健最低点开始保留的垂直比例 |
| `observation_profiles.<type>.vertical_max_fraction` | 1.0 | 从设备稳健最低点开始的观测上界比例 |
| `geometry.ground_z_m` | 0.0 | 预处理后的基准地面高度 |
| `geometry.ground_clearance_m` | 0.15 | 删除近地面点的高度余量 |
| `geometry.scan_stride` | 2 | 点云扫描采样步长 |
| `geometry.lower/upper_percentile` | 1 / 99.5 | 点云稳健包围范围 |
| `generation.candidate_stride_cells` | 4 | 候选停车位置采样间隔，当前为0.2 m |
| `generation.max_search_radius_m` | 30 | 每个设备轮廓外的预计算半径上界 |
| `generation.observation_model` | roi_conical_approach | 使用ROI圆锥—相机平面交圆环模型 |
| `generation.min_candidate_distance_m` | 0.2 | 数值上允许的最小相机水平距离 |
相机参数目前是常规初始值，不是特定硬件标定结果。真实部署时必须优先替换分辨率、视场角、安装高度/偏移和云台限位；不应通过修改设备类型停车距离弥补错误相机参数。

#### 5.4.10 静态点云局部视线通道遮挡检查

成像投影通过后，程序不对每个候选遍历6755万个点，而是把场景点云体素化并构建KD树。当前0.2 m体素将场景压缩为552728个占用体素，缓存为：

```text
outputs/<site>/planning/inspection_goal_regions/visibility_voxels.npz
```

缓存签名记录点云路径、文件大小、修改时间、体素尺寸和地图边界；点云或体素配置变化时自动失效，也可以使用 `--rebuild-visibility` 强制重建。

对一个候选相机位置 \(c\) 和ROI中心 \(q_{roi}\)，只沿线段：

\[
L(t)=c+t(q_{roi}-c),\qquad 0\le t\le1
\]

按 `ray_step_m` 采样，并查询采样点附近 `clearance_radius_m` 范围内的占用体素。该查询等价于检查相机和目标之间的局部圆柱/胶囊形视线通道，不访问与通道无关的场景点云。

每台设备查询时会从KD树中删除其自身二维轮廓柱内的体素，避免把目标设备表面误判为遮挡物；其他设备、支架和场景结构仍保留为潜在遮挡。相机附近和目标附近分别设置排除长度，减少机器人自身及终点表面的误判。

当前ROI圆锥模型只检查这一条视线通道；其他设备和场景结构仍会阻挡候选，目标设备自身的体素被排除。

主要配置为：

| 配置项 | 当前默认值 | 含义 |
|---|---:|---|
| `visibility.enabled` | true | 是否启用静态点云遮挡硬约束 |
| `visibility.voxel_size_m` | 0.2 | 场景占用体素尺寸 |
| `visibility.clearance_radius_m` | 0.2 | 视线通道基础半径 |
| `visibility.include_voxel_uncertainty` | true | 是否把半个体素对角线加入保守半径 |
| `visibility.ray_step_m` | 0.1 | 线段采样间隔 |
| `visibility.camera_exclusion_m` | 0.4 | 相机端忽略长度 |
| `visibility.target_exclusion_m` | 0.4 | 目标端忽略长度 |

该检查仍是静态点云近似：体素化和点云孔洞会产生误差，临时车辆、人员和施工物不在离线点云中。当前配置偏保守，后续应结合复核图、真实相机图像和规划成功率校准体素尺寸、通道半径与可见比例。

### 5.5 分层区域目标位姿 A*

传统 A* 的终点是固定栅格；改进算法的状态和目标集合为：

```text
s = (row, col, heading_bin)
终止条件：s in feasible_goal_poses_i
启发函数：当前位置到目标区域的最小二维距离
```

路径代价、转向代价和终端观测质量使用统一的米等价尺度。区域目标规划联合选择路径、停车位置、机身方向和云台姿态：

```text
(P*, g*) = argmin [J_path(P) + w_tilt J_tilt(g)]
subject to:
  rotated_robot_footprint(P) lies in S_safe
  endpoint(P) = g*
  g* in feasible_goal_poses_i
```

为避免在全站0.05 m栅格上同时展开全部航向层，当前采用分层搜索：第一阶段运行二维区域目标A*并计入终端视觉代价，第二阶段只在第一阶段路径周围的配置走廊内搜索 `(row,col,heading_bin)`。这样保留矩形足迹和最终姿态约束，同时显著降低内存和搜索时间。具体部件搜索、遮挡适应和最终拍摄仍由局部VLN负责。

当前实现：

```text
substation_vln/src/substation_vln/planning/improved_astar/pose_region_astar.py
substation_vln/tools/planning/run_region_goal_astar.py
```

#### 5.5.1 搜索状态与动作

改进A*状态为：

\[
s=(r,c,k),\qquad
\theta_k=\frac{2\pi k}{N_{\theta}}.
\]

当前机器狗按可横移的四足平台建模，动作包括：

```text
保持航向向4邻域或8邻域平移
原地向左旋转一个heading bin
原地向右旋转一个heading bin
```

平移后必须满足对应航向的 `pose_free_masks[k,r,c]=1`；对角移动还检查两个正交中间栅格，避免穿过障碍角点。旋转后也必须满足新航向矩形足迹可行。

#### 5.5.2 路径与终端代价

设一次平移的实际距离为 \(\Delta l\)，目标栅格语义代价为 \(C(n')\)，运动方向为 \(\theta_m\)，则平移代价为：

\[
J_{move}=\Delta l\left[
1+w_cC(n')+w_{lat}\left(1-|\cos(\theta_m-\theta_k)|\right)
\right].
\]

`lateral_motion_weight`使沿机身长轴方向移动略优于完全侧移，但不禁止四足机器人横移。一次原地转动一个航向格的代价为：

\[
J_{rotate}=w_{rot}.
\]

候选目标位姿 \(g\) 的终端代价为：

\[
J_{terminal}(g)=w_{tilt}J_{tilt}(g).
\]

总代价为：

\[
J(P,g)=\sum J_{move}+\sum J_{rotate}+J_{terminal}(g).
\]

路径长度和观测终端代价采用米等价尺度。默认 `tilt_cost_weight=15`；俯仰角偏离对应设备的推荐角度会直接影响A*终点选择。

#### 5.5.3 多终点启发函数与正确终止

将目标设备所有候选位置栅格化为目标位置掩码 \(M_G\)，使用距离变换得到当前栅格到目标区域的最小距离：

\[
D_G(n)=\rho\operatorname{DT}(\neg M_G)(n).
\]

启发函数为：

\[
h(n)=D_G(n)\left(1+w_cC_{min}\right).
\]

由于不同目标位姿的终端视觉代价不同，不能在第一次到达任意目标时立即停止。实现维护当前最优完整代价：

\[
J_{best}=g(g)+J_{terminal}(g),
\]

只有当开放列表最小优先级不再小于 \(J_{best}\) 时才结束，从而在非负终端代价下正确比较多个目标位姿。

#### 5.5.4 分层区域目标位姿搜索

在当前约3259 × 2595的0.05 m地图上直接展开16个航向层，会形成超过1.35亿个潜在状态。当前实现采用两阶段规划：

```text
阶段1：二维区域目标A*
  使用任意航向可容纳机器狗的位置作为二维可行空间
  使用所有目标位置和终端视觉代价
  输出一条全局粗路径

阶段2：走廊内位姿A*
  将粗路径按corridor_radius_m膨胀成搜索走廊
  只在走廊内展开(row,col,heading_bin)
  使用完整矩形足迹、旋转和平移代价
```

默认走廊半径为2 m。若姿态细化失败，自动扩大到4 m和8 m。该分层策略是计算加速方法，会限制第二阶段只在粗路径邻域内寻找姿态解；因此它是工程上的近似分层最优，而不是对全地图三维状态空间的严格全局最优证明。

#### 5.5.5 运行流程

完整执行顺序为：

```bash
python substation_vln/tools/planning/build_planning_map.py

python substation_vln/tools/planning/build_inspection_goal_regions.py

python substation_vln/tools/planning/run_region_goal_astar.py
```

第三个命令首先在命令行选择设备，再在窗口中选择机器狗起点。起始航向默认从 `run_region_goal_astar.yaml` 读取，也可使用命令行参数指定。非交互验证示例：

```bash
python substation_vln/tools/planning/run_region_goal_astar.py \
  --equipment 1 \
  --start-x 519100.0027 \
  --start-y 3344584.9911 \
  --start-yaw-deg 0 \
  --no-display
```

结果JSON记录：

```text
目标设备及候选数量
起点位置和航向
终点位置、机身航向、相机pan/tilt和ROI圆锥模型标识
逐节点(row,col,heading)、工程XY和yaw
路径长度、路径/视觉/总代价和扩展节点数
粗路径节点数、实际走廊半径和走廊栅格数
输入配置及输出文件位置
```

#### 5.5.6 测试可视化

运行时测试图使用自动缓存的干净2K点云俯视图作为底片，并叠加：

```text
深橙色：当前目标设备二维区域
绿色区域：满足碰撞、ROI圆锥和遮挡约束的候选停车区域
绿色饱和度：越饱和表示终端观测代价越低
红线/绿点/蓝点：最终路径、起点和选中终点
```

同一二维栅格若包含多个机身航向，显示其中最小终端代价；实际A*仍保留并比较全部三维状态 `(row,col,heading_bin)`。

#### 5.5.7 当前数据验证结果

当前合并标注包含10台巡视设备。采用ROI圆锥模型后，经过ROI中心静态视线过滤共保留58820个候选位姿，10台设备全部保持非空。其中两台主变分别保留18258和19852个候选位姿，其解析半径范围分别约为3.708--17.643 m和3.705--17.630 m。

一次实际链路测试得到：

```text
目标设备：1#bianyaqi
路径长度：25.603 m
终点机身航向：292.5 deg
相机pan：93.3 deg
相机tilt：29.8 deg
俯仰角代价：0.022
加权终端观测代价：0.336
ROI中心视线：无遮挡
```

该结果验证了世界坐标—栅格坐标、ROI圆锥圆环反算、单点局部视线、切向停车方向、矩形碰撞和分层A*的完整数据链路。当前结果只代表现有点云、角度定义和测试起点的工程验证，不应直接作为论文统计结论。

#### 5.5.8 A*参数位置

配置文件为：

```text
substation_vln/configs/tools/planning/run_region_goal_astar.yaml
```

| 配置项 | 当前默认值 | 实现含义 |
|---|---:|---|
| `start.yaw_deg` | 0 | 交互选择起点时的默认初始机身方向 |
| `hierarchical.corridor_radius_m` | 2 | 首次姿态细化走廊半径 |
| `hierarchical.max_corridor_radius_m` | 8 | 自动回退允许的最大走廊半径 |
| `astar.cost_weight` | 1.0 | 栅格语义代价权重 |
| `astar.heuristic_weight` | 1.0 | A*启发函数权重；1.0保持标准A*形式 |
| `astar.rotation_cost_per_bin` | 0.35 | 原地旋转一个航向格的米等价代价 |
| `astar.lateral_motion_weight` | 0.25 | 侧向移动附加代价 |
| `astar.tilt_cost_weight` | 15 | 终端俯仰偏好的米等价权重 |
| `astar.allow_diagonal` | true | 是否允许8邻域平移 |

通用语义地图参数仍位于：

```text
substation_vln/configs/tools/planning/build_planning_map_erfeishan.yaml
```

其中道路、优先路径、狭窄空间和障碍净空只改变路径软代价；规划边界、障碍物、设备区域和矩形足迹构成硬约束。

### 5.6 可选的任务与现场环境软代价

效率、标准、保守三种模式可以作为固定权重基线或典型工作点，但不作为当前第一阶段的必要模块。后续若项目指标需要，可研究连续上下文到代价权重的映射：

```text
w = f(z)
z = [任务紧急程度, 环境风险, 定位可靠度, 续航压力, ...]
```

任务解析模型可以从自然语言巡视指令中提取：

```text
目标设备
巡视部件
任务类型
时间要求
文本中的紧急语义
```

最终紧急程度不能只由语言模型自由决定，还应融合站内告警等级、人工指定等级和设备任务规则，并由确定性规则模块校验。

天气预报只作为环境先验。实际环境状态应融合：

```text
天气接口：降雨/降雪概率、温度、风速和能见度
站内传感器：温湿度、降雨、路面温度等定量信息
机器人现场视觉：积水、积雪、疑似结冰、低能见度和临时占用
```

湿度等不可可靠从RGB图像直接测量的变量必须来自传感器。积水、积雪和临时障碍等局部现象还应形成空间风险图，而不是全部压缩为一个全局模式。所有自适应结果只改变软代价，不能放松 `S_safe`。该部分暂不加入当前代码主链路。

### 5.7 全局规划与局部 VLN 的职责边界

```text
全局阶段：
  根据结构化任务查找目标设备区域
  生成或读取相机可行目标位姿区域
  使用矩形足迹和二维硬安全空间筛选终端位姿
  使用区域目标位姿A*完成长距离运动
  进入目标位姿区域后移交局部控制权

在线局部阶段：
  接收目标设备、待拍摄部件和可选视角要求
  获取实时图像、深度、机器人位姿和云台姿态
  在安全动作集合内搜索指定设备部件
  调整底盘位置与相机朝向并输出CAPTURE
  保存图像和采集元数据后结束当前拍摄任务
```

VLN不判断设备是否正常、异常或故障。“任务完成”只表示相机拍摄动作成功执行并保存文件。若局部运动被临时障碍完全阻断，局部模块可以返回 `REQUEST_REPLAN`，由全局规划重新选择目标位姿或路径；安全控制器始终有权拒绝VLN的不安全动作。

### 5.8 底盘—云台—拍摄分解动作空间

若机器人使用可转动云台相机，动作空间分为三部分：

```text
A_base   = {前进, 后退, 左转, 右转, 保持}
A_camera = {左看, 右看, 上看, 下看, 回中, 保持}
A_task   = {无操作, 拍摄, 完成, 请求全局重规划}
```

时刻 `t` 的动作表示为：

```text
a_t = (a_t_base, a_t_camera, a_t_task)
```

分解动作避免把“前进并向上看”等组合全部定义为独立类别。初期仿真可采用分阶段执行：先移动到设备附近，再停止底盘并控制云台搜索，最后输出 `CAPTURE`。真实机器人中，VLN输出离散动作或局部航点，底盘速度和碰撞约束仍由确定性控制器执行。

### 5.9 设备知识增强

为了提高不同设备类型和不同视角下的局部拍摄效率，建立变电站设备知识库：

```text
设备台账与唯一名称
典型设计资料和巡视规程
设备整体与关键部件的多角度参考图片
设备类型—部件名称—视觉特征的对应关系
```

设备资料不应简单拼接后一次性输入模型。更合适的方式是根据当前 `equipment_type` 和 `inspection_part` 检索相关文本和参考图片，再作为局部VLN的条件信息。论文应通过去除知识库的消融实验，验证其是否减少局部运动步数、重复搜索和拍摄失败。

当前DeepSeek接口仅用于文本任务解析，不假设其能够承担连续图像VLN。市场上存在可远程调用的通用多模态模型，但没有可直接替代机器人局部控制器的标准付费VLN API；后续可选择云GPU部署开源VLN策略，或使用在线多模态模型做低频语义决策、本地控制器负责高频安全执行。

### 5.10 Baseline A*

论文中的 baseline A* 应保持为传统栅格 A*：

```text
输入：free_space_mask、起点、终点
状态：(row, col)
邻域：4 邻域或 8 邻域
实际代价：直线步长 1，对角步长 sqrt(2)
启发函数：Manhattan、Euclidean 或 Octile distance
终点：人工固定巡视点或固定安全停靠点
不使用：相机目标位姿区域、preferred road、narrow space、preferred path 和障碍物软排斥
```

这样可以量化“固定点终点”与“相机可行区域终点”的差异，并为后续分层VLN系统提供传统全局规划基线。

## 6. 对比与消融实验

### 6.1 算法组

建议至少设置：

```text
B0  人工固定巡视点 + 传统A*
B1  从相机可行区域预先固定一个终点 + 传统A*
B2  相机可行位置区域 + 二维区域目标A*（不规划机身航向）
B3  相机可行位姿区域 + 矩形足迹分层A* + 固定语义代价
L0  B3 + 不使用设备知识的通用局部VLN
M1  B3 + 设备知识增强局部VLN（完整方法）
```

可选外部全局规划基线包括Dijkstra、JPS和Theta*。全局规划与局部VLN应分阶段统计，不能把网络推理时间和静态A*搜索时间混成一个指标。若后续实现环境上下文软代价，再单独增加对应消融组。

### 6.2 测试任务

```text
使用多个设备相机可行位姿区域和多个二维可行起点构造任务
增加规划边界内随机可行起点
覆盖多个分离目标位姿区域、窄通道和道路绕行场景
构造固定巡视点被临时障碍占用但设备仍可从其他方向接近的案例
构造设备型号或关键部件相对位置发生变化的跨设备测试
使用不同表达方式的操作票和自然语言拍摄指令
测试底盘移动与云台左右、上下调整的组合任务
建议形成不少于 100 组有效规划任务
```

### 6.3 评价指标

```text
任务解析设备名称/部件/顺序的精确匹配率
全局路径长度、规划时间、扩展节点数和峰值内存
目标位姿数量、位置覆盖面积与有效连通分量数量
固定巡视点失效时的全局规划成功率
最小/平均障碍物距离、狭窄空间内路径长度和道路内路径比例
局部VLN拍摄任务成功率、局部运动步数和完成时间
底盘移动次数、云台动作次数、重复搜索次数和全局重规划次数
新设备实例、新起点、临时障碍和不同语言表达下的泛化成功率
单台设备人工标注时间、人工固定点数量和后期维护次数
```

运行时系统只记录是否成功执行拍摄；实验评测可以使用人工真值检查是否拍摄了正确设备和部件，但该检查不属于VLN在线故障判断。所有指标应报告均值、标准差和显著性检验，不应只展示少数视觉效果较好的路径。

### 6.4 参数敏感性

需要重点分析：

```text
地图分辨率：0.05 m、0.10 m
baseline障碍物膨胀半径；improved A*矩形尺寸与软排斥半径/权重
ROI高度比例、各设备俯仰范围与推荐俯仰角
机器狗矩形尺寸、安全边距与航向离散数
preferred_path_sigma_m
preferred_path_alpha
preferred_road_cost
狭窄空间、障碍净空和方向代价权重
底盘平移/旋转步长与云台pan/tilt步长
局部VLN最大动作步数和停止条件
设备知识检索数量与参考视图数量
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
9. Anderson et al., “Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments,” CVPR, 2018.
10. Hong et al., “VLN BERT: A Recurrent Vision-and-Language BERT for Navigation,” CVPR, 2021.
11. Chen et al., “History Aware Multimodal Transformer for Vision-and-Language Navigation,” NeurIPS, 2021.
12. Chen et al., “Think Global, Act Local: Dual-Scale Graph Transformer for Vision-and-Language Navigation,” CVPR, 2022.
13. Zheng et al., “Towards Generalizable Embodied Navigation via Large Language Model,” CVPR, 2024.

## 8. 后续工作

后续主要内容将围绕以下方向继续完善：

```text
已完成：标注当前10台巡视设备的equipment_region
已完成：从设备点云和相机参数生成逐设备可行目标位姿
已完成：矩形足迹、切向航向、区域距离启发图和分层位姿A*
下一步1：使用真实相机内参、安装外参和云台限位替换常规默认参数
下一步2：建立设备台账并校验DeepSeek任务中的设备名称与地图语义
下一步3：在Habitat-GS中实现底盘、云台、CAPTURE和REQUEST_REPLAN动作接口
下一步4：整理设备典型设计资料和多角度图片，构建设备知识检索数据
下一步5：实现不含故障判断的局部VLN搜索与拍摄策略
下一步6：对比固定点A*、二维区域A*、位姿区域A*、通用VLN和知识增强VLN
下一步7：使用真实图像校准静态点云遮挡阈值，并根据实验需要加入环境软代价和多源风险融合
```

当前已经完成点云坐标预处理、地面归零、Gaussian配准工具链、二维语义标注、10台巡视设备区域标注、规划地图构建、固定终点A*基线、ROI圆锥目标位姿预计算、静态点云局部视线遮挡、分层区域目标位姿A*以及DeepSeek文本任务解析接口。标注工具为每次JSON标注生成同目录同名2K复核缩略图。下一阶段应先用真实硬件参数和现场图像复核终端位姿与遮挡阈值，再实现Habitat-GS中的局部VLN控制与设备知识增强。
