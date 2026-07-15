# 二维语义代价 A* 技术方案

本文档说明当前项目中二维 A* 的数据来源、地图构建、代价函数、搜索过程、参数位置、运行步骤和验证方法。本文描述的是当前代码已经采用的实现，不包含后续可行巡视区域生成和区域目标 A*。

## 1. 目标与适用范围

当前二维规划模块用于在变电站静态语义地图中，从机器人起点规划到人工标注的巡视停靠点。算法遵循以下原则：

```text
规划边界和膨胀障碍物是硬约束
道路、优先路径、狭窄空间和障碍净空是软代价
A* 始终在唯一的 free_space_mask 内搜索
软代价改变路径偏好，但不能突破硬安全空间
```

当前实现主要用于：

- 验证点云正射图、语义标注、世界坐标和栅格坐标之间的完整链路；
- 生成固定巡视点的二维全局路径；
- 为后续区域目标 A* 和上下文自适应代价提供二维基线。

当前算法假设地图静态，不直接处理实时动态障碍。在线避障和最终观测位姿调整由后续局部规划或 VLN 模块负责。

## 2. 软件模块

二维规划相关文件如下：

```text
substation_vln/src/substation_vln/planning/common/grid.py
  GridSpec 与世界坐标/栅格坐标转换

substation_vln/src/substation_vln/planning/common/base_map.py
  从合并标注生成基础语义 mask

substation_vln/src/substation_vln/planning/common/derived_map.py
  障碍膨胀、距离场、软代价和 cost_map

substation_vln/src/substation_vln/planning/common/io.py
  地图保存和可视化

substation_vln/src/substation_vln/planning/astar/astar.py
  二维栅格 A* 核心搜索

substation_vln/tools/planning/build_planning_map.py
  规划地图构建入口

substation_vln/tools/planning/run_baseline_astar.py
  起点选择、巡视点选择、A*执行和结果保存入口
```

站点参数文件：

```text
substation_vln/configs/tools/planning/build_planning_map_erfeishan.yaml
substation_vln/configs/tools/planning/run_baseline_astar_erfeishan.yaml
```

## 3. 输入数据

### 3.1 合并标注

默认输入：

```text
substation_vln/data/annotations/220kv_erfeishan/annotations_merged.json
```

二维规划使用以下类别：

| 类别 | 几何 | 规划含义 |
|---|---|---|
| `planning_boundary` | 多边形 | 机器人允许活动的最外层范围 |
| `obstacle` | 多边形或圆形 | 不可通行障碍物 |
| `preferred_road` | 多边形 | 低代价道路区域 |
| `narrow_space` | 多边形或圆形 | 可通行但具有附加代价的狭窄区域 |
| `preferred_path` | 折线 | 路径吸引中心线 |
| `patrol_point` | 有向点 | 当前固定终点及观察方向元数据 |

标注文件中的几何同时保存像素坐标和工程坐标。规划地图只使用工程 XY 坐标。

### 3.2 坐标约定

工程坐标点记为：

\[
\mathbf p=(x,y).
\]

栅格坐标记为：

\[
\mathbf n=(r,c),
\]

其中 `r` 为自上向下增加的行，`c` 为自左向右增加的列。设地图范围为

\[
[x_{\min},x_{\max}]\times[y_{\min},y_{\max}],
\]

分辨率为 \(\rho\)，单位为米/像素，则世界坐标到栅格坐标为：

\[
c=\left\lfloor\frac{x-x_{\min}}{\rho}\right\rfloor,
\qquad
r=\left\lfloor\frac{y_{\max}-y}{\rho}\right\rfloor.
\]

栅格中心到世界坐标为：

\[
x=x_{\min}+(c+0.5)\rho,
\qquad
y=y_{\max}-(r+0.5)\rho.
\]

对应实现位于 `GridSpec.xy_to_grid()` 和 `GridSpec.grid_to_xy()`。

## 4. 基础语义地图构建

### 4.1 规划栅格范围

规划边界包围盒向外增加 `bounds_padding_m`，栅格尺寸为：

\[
W=\left\lceil\frac{x_{\max}-x_{\min}}{\rho}\right\rceil,
\qquad
H=\left\lceil\frac{y_{\max}-y_{\min}}{\rho}\right\rceil.
\]

当前二妃山地图使用：

```text
resolution_m = 0.05 m/pixel
bounds_padding_m = 1.0 m
```

### 4.2 基础 mask

所有基础图层均为 \(H\times W\) 的二值数组：

```text
boundary_mask
obstacle_mask
preferred_road_mask
narrow_space_mask
preferred_path_mask
```

多边形通过 OpenCV `fillPoly` 栅格化，圆形通过 `circle` 填充，优先路径折线按 `preferred_path_width_m` 绘制为具有宽度的带状区域。

基础图层经过边界裁剪：

\[
M_{obs}=M_{obs}^{raw}\cap M_{boundary},
\]

\[
M_{road}=M_{road}^{raw}\cap M_{boundary}\cap\neg M_{obs},
\]

\[
M_{narrow}=M_{narrow}^{raw}\cap M_{boundary}\cap\neg M_{obs}.
\]

## 5. 硬安全空间

### 5.1 障碍距离场

对障碍物 mask 计算二维欧氏距离变换：

\[
D_{obs}(x)=\rho\,\operatorname{DT}(\neg M_{obs})(x),
\]

其中 \(D_{obs}(x)\) 表示栅格中心到最近障碍物的近似距离，单位为米。

### 5.2 障碍膨胀

设膨胀半径为 \(r_{inflate}\)，膨胀障碍物定义为：

\[
M_{inflated}(x)=
\begin{cases}
1,&D_{obs}(x)\le r_{inflate},\\
0,&D_{obs}(x)>r_{inflate}.
\end{cases}
\]

自由空间为：

\[
M_{free}=M_{boundary}\cap\neg M_{inflated}.
\]

当前阶段假设障碍物标注和膨胀半径已经包含机器人本体尺寸及所需安全余量。A* 不允许进入 `free_space_mask == 0` 的栅格。

## 6. 语义软代价地图

### 6.1 基础代价与道路代价

对自由空间赋基础代价：

\[
C(x)=C_{base},\qquad x\in M_{free}.
\]

道路区域覆盖为较低代价：

\[
C(x)=C_{road},\qquad x\in M_{road}\cap M_{free}.
\]

通常应满足：

\[
0<C_{road}<C_{base}.
\]

### 6.2 优先路径吸引

设栅格到优先路径的距离为 \(D_{path}(x)\)，路径吸引场为：

\[
A_{path}(x)=
\exp\left(-\frac{D_{path}(x)^2}{2\sigma_{path}^2}\right).
\]

将路径吸引作为代价减项：

\[
C(x)\leftarrow C(x)-\alpha_{path}A_{path}(x).
\]

`preferred_path_sigma_m` 决定吸引场宽度，`preferred_path_alpha` 决定吸引强度。如果没有标注优先路径，吸引场为零。

当前 `preferred_path` 虽可保存有向或无向属性，但本版二维 A* 只使用其空间吸引，不使用方向约束。

### 6.3 狭窄空间惩罚

狭窄空间保持可通行，在其内部增加固定附加代价：

\[
C(x)\leftarrow C(x)+\lambda_{narrow},
\qquad x\in M_{narrow}\cap M_{free}.
\]

当绕行代价较低时，A*会主动避开狭窄空间；当狭窄空间是唯一可达通道或绕行过长时，仍允许从中通过。

### 6.4 障碍物软排斥

设软排斥作用半径为 \(r_{rep}\)，权重为 \(w_{rep}\)，定义：

\[
R_{obs}(x)=w_{rep}
\left[
\max\left(0,\frac{r_{rep}-D_{obs}(x)}{r_{rep}}\right)
\right]^2.
\]

更新代价：

\[
C(x)\leftarrow C(x)+R_{obs}(x).
\]

障碍膨胀负责硬碰撞约束，软排斥用于在可通行空间内进一步增加净空。

### 6.5 最终代价

对自由空间代价设置下限：

\[
C(x)\leftarrow\max(C(x),C_{min}).
\]

对不可通行空间设置：

\[
C(x)=+\infty,\qquad x\notin M_{free}.
\]

结合当前代码执行顺序，最终代价为：

\[
C(x)=\max\left(
C_{region}(x)
-\alpha_{path}A_{path}(x)
\lambda_{narrow}M_{narrow}(x)
R_{obs}(x),
C_{min}
\right),
\]

其中：

\[
C_{region}(x)=
\begin{cases}
C_{road},&x\in M_{road},\\
C_{base},&x\notin M_{road}.
\end{cases}
\]

## 7. 二维 A* 搜索

### 7.1 状态与邻域

当前状态只包含二维栅格位置：

\[
n=(r,c).
\]

支持4邻域和8邻域。8邻域的直线步长为1，对角步长为 \(\sqrt{2}\)：

\[
\ell(u,v)=
\begin{cases}
1,&\text{水平或垂直移动},\\
\sqrt{2},&\text{对角移动}.
\end{cases}
\]

### 7.2 边代价

从当前节点 \(u\) 移动到相邻节点 \(v\) 的代价为：

\[
c(u,v)=\ell(u,v)
\left[1+w_c\max(C(v),\varepsilon)\right],
\]

其中：

- \(w_c\) 对应 `cost_weight`；
- \(\varepsilon\) 对应 `min_traversal_cost`；
- \(C(v)\) 为目标栅格的语义代价。

当前搜索累计的是栅格尺度代价。最终输出路径长度时，再将栅格中心转换为世界坐标并以米计算几何长度。

### 7.3 启发函数

当前启发函数为栅格坐标中的欧氏距离：

\[
h(n,g)=\sqrt{(r_n-r_g)^2+(c_n-c_g)^2}.
\]

节点优先级为：

\[
f(n)=g(n)+w_hh(n,g),
\]

其中 `heuristic_weight` 对应 \(w_h\)。当前默认值为1。

若论文需要严格讨论最优性，应证明启发函数不高估最小剩余代价。`heuristic_weight > 1` 会转为加权 A*，通常减少扩展节点，但不再保证原代价定义下的最优路径。

### 7.4 搜索终止与路径恢复

当前终点为一个固定巡视点栅格。当弹出的最小优先级节点等于终点时终止搜索，通过 `came_from` 反向恢复路径。

起点和终点必须同时满足：

```text
位于地图范围内
free_space_mask == 1
cost_map为有限值
```

否则程序直接报告该点不可通行。固定点仅用于标准A*基线实验；主方法不再把人工固定点作为设备巡视终点。

## 8. 参数设置位置

### 8.1 地图与语义代价参数

文件：

```text
substation_vln/configs/tools/planning/build_planning_map_erfeishan.yaml
```

| YAML路径 | 含义 | 当前值 |
|---|---|---:|
| `paths.annotation` | 合并标注输入 | `annotations_merged.json` |
| `paths.output_dir` | 规划地图输出目录 | `planning/maps` |
| `base_map.resolution_m` | 地图分辨率，m/pixel | `0.05` |
| `base_map.bounds_padding_m` | 规划边界包围盒外扩 | `1.0` |
| `base_map.preferred_path_width_m` | 优先路径栅格化宽度 | `0.5` |
| `derived_map.obstacle_inflation_radius_m` | 障碍膨胀半径 | `0.4` |
| `derived_map.preferred_path_sigma_m` | 优先路径吸引场尺度 | `1.5` |
| `derived_map.preferred_path_alpha` | 优先路径吸引强度 | `0.3` |
| `derived_map.narrow_space_penalty` | 狭窄空间附加代价 | `1.5` |
| `derived_map.obstacle_repulsion_radius_m` | 障碍软排斥半径 | `1.0` |
| `derived_map.obstacle_repulsion_weight` | 障碍软排斥权重 | `0.8` |
| `derived_map.base_cost` | 普通自由空间基础代价 | `1.0` |
| `derived_map.preferred_road_cost` | 道路区域基础代价 | `0.6` |
| `derived_map.min_cost` | 最小栅格代价 | `0.2` |

调整这些参数后必须重新运行 `build_planning_map.py`，因为它们已经固化进 `planning_map.npz` 的 `cost_map`。

### 8.2 A*搜索参数

文件：

```text
substation_vln/configs/tools/planning/run_baseline_astar_erfeishan.yaml
```

| YAML路径 | 含义 | 当前值 |
|---|---|---:|
| `astar.connectivity` | 4或8邻域 | `8` |
| `astar.cost_weight` | 语义代价在边代价中的权重 | `1.0` |
| `astar.heuristic_weight` | 启发函数权重 | `1.0` |
| `astar.min_traversal_cost` | 栅格代价数值下限 | `1e-6` |

显示窗口和路径绘制参数位于同一文件的 `display` 节点。修改A*搜索参数不需要重建规划地图，只需重新运行 `run_baseline_astar.py`。

### 8.3 参数职责边界

```text
build_planning_map_erfeishan.yaml
  决定地图是什么、哪里可通行、不同语义区域代价是多少

run_baseline_astar_erfeishan.yaml
  决定A*如何搜索以及如何显示结果
```

障碍膨胀半径属于地图安全参数，不应根据任务紧急程度随意降低。道路、狭窄空间和优先路径权重属于软偏好参数，后续可以由上下文自适应模块调节。

## 9. 实施步骤

### 9.1 合并标注

```bash
conda run -n habitat-gs python \
  substation_vln/tools/annotation/merge_annotation_files.py \
  --config substation_vln/configs/tools/annotation/merge_annotation_files_erfeishan.yaml
```

检查：

```text
substation_vln/outputs/220kv_erfeishan/annotation_reviews/annotations_merged_review.png
```

### 9.2 构建规划地图

```bash
conda run -n habitat-gs python \
  substation_vln/tools/planning/build_planning_map.py \
  --config substation_vln/configs/tools/planning/build_planning_map_erfeishan.yaml
```

检查：

```text
boundary_mask.png
obstacle_mask.png
inflated_obstacle_mask.png
free_space_mask.png
preferred_road_mask.png
narrow_space_mask.png
preferred_path_mask.png
cost_map.png
planning_overlay.png
```

### 9.3 运行交互式 A*

```bash
conda run -n habitat-gs python \
  substation_vln/tools/planning/run_baseline_astar.py \
  --config substation_vln/configs/tools/planning/run_baseline_astar_erfeishan.yaml
```

交互过程：

```text
1. 在自由空间中左键选择起点
2. 按Enter确认起点
3. 在命令行输入巡视点编号
4. 输入Y确认目标
5. A*搜索并保存路径JSON和叠加图
6. 在结果窗口按任意键关闭
```

### 9.4 输出

地图输出目录：

```text
substation_vln/outputs/220kv_erfeishan/planning/maps/
```

路径输出目录：

```text
substation_vln/outputs/220kv_erfeishan/planning/baseline_astar/
```

路径JSON记录：

```text
起点和终点栅格/世界坐标
A*参数
路径栅格序列
路径世界坐标序列
路径长度
累计总代价
扩展节点数
输出文件位置
```

## 10. 验证要求

每次重新标注或调整地图参数后至少检查：

1. `free_space_mask` 是否为预期的连通区域；
2. 障碍膨胀是否阻断本应连通的道路；
3. `narrow_space_mask` 是否仍包含在自由空间中；
4. 狭窄空间代价是否高于相同条件下的普通区域；
5. 巡视点是否位于自由空间；
6. 短距离、长距离、绕行和必须经过狭窄空间的任务是否都能合理规划；
7. 路径是否存在穿越障碍物或地图边界的情况；
8. 参数、地图metadata和路径结果是否一同保存，保证实验可复现。

建议报告以下路径指标：

```text
路径长度
累计规划代价
扩展节点数和搜索时间
道路内路径比例
狭窄空间内路径长度
最小/平均障碍净空
规划成功率
```

## 11. 当前限制与后续扩展

当前实现存在以下边界：

- 固定单栅格终点可能落入膨胀障碍，后续将替换为可行巡视区域目标；
- 8邻域对角移动尚未显式禁止从两个障碍角之间穿越，需要补充防止 corner cutting 的检查；
- 状态不包含机器人朝向，尚未建模转弯半径和连续运动学；
- `preferred_path` 的有向属性尚未进入A*边代价；
- 当前 `cost_map` 在地图构建时固定，尚未实现任务与环境上下文驱动的运行时代价生成；
- 5厘米大地图上的纯Python字典A*在长距离任务中扩展节点较多，后续需要优化内存结构或采用多分辨率搜索。

后续区域目标 A* 将保持同一硬安全空间和语义代价定义，只把固定终点替换为 `inspection_approach_region_mask`，并使用到目标区域的距离变换作为启发函数。
