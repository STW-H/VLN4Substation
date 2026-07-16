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

完成坐标对齐后进入二维标注流程。当前工具已经支持地图语义、设备占地区域和人工可停靠范围，并统一保存像素坐标与工程XY坐标。

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

下一步将在规划地图中增加逐设备的区域数据：

```text
equipment_region                  设备占地范围及设备名称/类型
inspection_approach_region_mask  A*与局部VLN的逐设备交接区域
```

该扩展尚未在当前 `build_planning_map.py` 中实现；现有规划地图和 `patrol_points.json` 继续服务于固定终点A*基线。

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

### 5.4 设备接近区域生成

离线阶段不再从三维目标点计算精细可见域，而是生成A*与局部VLN的设备接近区域。设备占地区域记为 \(E_i\)，最小安全距离和最大交接距离分别为 \(d_{\min}\) 与 \(d_{\max}\)：

```text
G_buffer_i = Buffer(E_i, d_max) - Buffer(E_i, d_min)
G_approach_i = G_buffer_i & free_space_mask
```

若设备周围存在仅允许单侧进入等特殊约束，则直接使用人工标注的 `inspection_approach_region`，并同样与 `free_space_mask` 求交。该区域只表示全局导航允许交接的位置，不保证当前位置已经能拍清指定部件；具体搜索、视角调整和拍摄由局部VLN负责。

### 5.5 区域目标 A*

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
  生成或读取设备接近区域
  与二维硬安全空间求交
  使用区域目标A*完成长距离运动
  进入接近区域后移交局部控制权

在线局部阶段：
  接收目标设备、待拍摄部件和可选视角要求
  获取实时图像、深度、机器人位姿和云台姿态
  在安全动作集合内搜索指定设备部件
  调整底盘位置与相机朝向并输出CAPTURE
  保存图像和采集元数据后结束当前拍摄任务
```

VLN不判断设备是否正常、异常或故障。“任务完成”只表示相机拍摄动作成功执行并保存文件。若局部运动被临时障碍完全阻断，局部模块可以返回 `REQUEST_REPLAN`，由全局规划重新选择接近区域或路径；安全控制器始终有权拒绝VLN的不安全动作。

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
不使用：设备接近区域、preferred road、narrow space、preferred path 和障碍物软排斥
```

这样可以量化“固定点终点”与“设备接近区域终点”的差异，并为后续分层VLN系统提供传统全局规划基线。

## 6. 对比与消融实验

### 6.1 算法组

建议至少设置：

```text
B0  人工固定巡视点 + 传统A*
B1  从设备接近区域中预先固定一个终点 + 传统A*
B2  设备区域外扩 + 区域目标A*
B3  人工/自动接近区域 + 区域目标A* + 固定语义代价
L0  B3 + 不使用设备知识的通用局部VLN
M1  B3 + 设备知识增强局部VLN（完整方法）
```

可选外部全局规划基线包括Dijkstra、JPS和Theta*。全局规划与局部VLN应分阶段统计，不能把网络推理时间和静态A*搜索时间混成一个指标。若后续实现环境上下文软代价，再单独增加对应消融组。

### 6.2 测试任务

```text
使用多个设备可停靠范围和多个二维可行起点构造任务
增加规划边界内随机可行起点
覆盖多个分离接近区域、窄通道和道路绕行场景
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
设备接近区域面积与有效连通分量数量
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
障碍物膨胀半径：根据机器人半宽和安全余量设置
设备接近区域最小距离 d_min 与最大距离 d_max
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
1. 标注巡视目标设备的equipment_region及少量特殊设备的人工接近区域
2. 实现设备区域外扩、自由空间求交和逐设备approach_region_mask
3. 实现区域目标A*、目标区域距离启发图及对应技术方案
4. 建立设备台账并校验DeepSeek任务中的设备名称与地图语义
5. 在Habitat-GS中实现底盘、云台、CAPTURE和REQUEST_REPLAN动作接口
6. 整理设备典型设计资料和多角度图片，构建设备知识检索数据
7. 实现不含故障判断的局部VLN搜索与拍摄策略
8. 对比固定点A*、区域目标A*、通用VLN和知识增强VLN
9. 根据项目进度再决定是否加入环境上下文软代价和多源风险融合
```

当前已经完成点云坐标预处理、地面归零、Gaussian配准工具链、二维语义标注、规划地图构建、固定终点A*基线以及DeepSeek文本任务解析接口。标注工具已经支持设备占地区域、人工可停靠范围和同目录同名2K复核缩略图。下一阶段应先标注目标设备并实现设备接近区域与区域目标A*；完成全局交接链路后，再逐步实现局部VLN和设备知识增强。
