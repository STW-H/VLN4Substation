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

VLN 的主要价值放在设备接近区域内的局部拍摄，而不是替代传统全局导航：

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
5. 由设备区域向外扩展或人工绘制A*与VLN的设备接近区域
6. 区域目标A*联合选择接近区域内的交接位置和长距离全局路径
7. 到达接近区域后，局部VLN控制底盘和云台搜索指定部件并触发拍摄
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
  annotate.py                           # 二维语义、设备区域与可停靠范围标注
  merge_annotation_files.py             # 多次标注结果合并

tasks/
  parse_inspection_instruction.py       # 远程DeepSeek解析操作票/自然语言拍摄任务

visualization/
  view_pointcloud.py                    # 查看完整点云和普通点云
  view_gaussian.py                      # 使用 Habitat-GS 查看/渲染高斯

planning/
  build_planning_map.py                 # 从合并标注构建规划地图
  run_baseline_astar.py                 # 二维 A* 基线规划与验证
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

### 3.3 设备区域与可停靠范围标注

当前主流程不再逐点标注三维巡视目标。离线地图只描述设备占地区域及A*与局部VLN之间的交接范围；具体部件搜索、视角调整和拍摄由局部VLN完成。

标注数据使用统一的时间戳文件：

```text
data/annotations/<site>/sessions/annotation_<timestamp>.json  # 矢量标注主数据
data/annotations/<site>/sessions/annotation_<timestamp>.png   # 同名2K复核缩略图
```

每次标注必须同时生成JSON和最长边2048像素的同名PNG；缩略图只用于快速复核，规划始终读取JSON。多次会话合并后对应生成 `annotations_merged.json` 和 `annotations_merged.png`。

支持两种数据表达：

```text
equipment_region
  人工标注设备占地轮廓、设备名称和类型
  设备名称应与操作票、设备台账和自然语言指令中的唯一名称一致
  保存 approach_region.min_distance_m / max_distance_m
  后续按 Buffer(E,d_max)-Buffer(E,d_min) 自动生成交接区域

inspection_approach_region
  直接人工绘制设备的可停靠范围
  适合设备周围空间不规则或只允许从特定方向接近的情况
```

无论采用哪种方式，生成的区域都必须与二维自由空间求交。该区域是全局A*与局部VLN的控制权交接范围，不等同于保证部件清晰可见的精确三维观测空间。A*只负责到达该区域，局部VLN负责在区域内寻找具体拍摄位置。

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
1. 以设备接近区域代替唯一固定巡视点的全局—局部交接机制
2. 操作票/自然语言到地图设备及局部拍摄任务的结构化语义连接
3. 融合设备多角度资料的局部VLN拍摄策略及跨设备适应性
```

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

### 5.3 设备接近区域生成

离线阶段不再从三维目标点计算精细可见域，而是生成A*与局部VLN的设备接近区域。设备占地区域记为 (E_i)，最小安全距离和最大交接距离分别为 (d_{min}) 与 (d_{max})：

```text
G_buffer_i = Buffer(E_i, d_max) - Buffer(E_i, d_min)
G_approach_i = G_buffer_i & free_space_mask
```

若设备周围存在仅允许单侧进入等特殊约束，则直接使用人工标注的 `inspection_approach_region`，并同样与 `free_space_mask` 求交。该区域只表示全局导航允许交接的位置，不保证当前位置已经能拍清指定部件；具体搜索、视角调整和拍摄由局部VLN负责。

### 5.4 区域目标 A*

传统 A* 的终点是固定栅格；本文将 `inspection_approach_region_mask_i` 作为目标区域：

```text
终止条件：current cell in inspection_approach_region_mask_i
启发函数：当前栅格到目标区域的最小二维距离
```

目标区域距离可预先使用二维距离变换生成。区域目标规划联合选择停靠位置与路径：

```text
(P*, g*) = argmin J(P | z)
subject to:
  P lies in S_safe
  endpoint(P) = g*
  g* in G_approach_i
```

离线阶段不预测目标在真实图像中的清晰度、居中程度或检测置信度。只要终点属于安全接近区域，即可将控制权交给局部VLN。

### 5.5 任务与现场环境驱动的代价自适应

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

### 5.6 全局规划与局部 VLN 的职责边界

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

### 5.7 Baseline A*

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
使用多个设备可停靠范围和多个二维可行起点构造任务
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
标注设备区域或人工可停靠范围
实现设备区域外扩并与二维自由空间求交
实现区域目标 A* 及目标区域距离启发图
实现固定语义代价与上下文自适应代价的对比方法
生成 preferred path 切向量场
完成消融实验和参数敏感性分析
实现局部 VLN 主动观测和安全动作接口
实现 VLM 指令解析及现场环境多源融合接口
仿真验证流程
```

当前已经完成点云坐标预处理、地面归零、Gaussian 配准工具链、语义标注、规划地图构建和交互式 A* 链路验证。下一阶段首先在新坐标系中重新生成二维标注和规划地图，然后实现三维点云可见区域与区域目标 A*。语义风险自适应和局部 VLN 在该离线主链路验证完成后接入。
