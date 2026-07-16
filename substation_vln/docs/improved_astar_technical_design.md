# 改进式区域目标位姿 A* 详细设计

## 1. 设计目标

普通 A* 将机器人简化为质点，并要求到达一个固定目标栅格。变电站巡视不适合这种假设：设备周围通常存在多个可观察位置，机器狗具有不可忽略的矩形尺寸，停靠方向会影响后续移动，云台姿态和视线遮挡也会影响拍摄质量。

本算法解决以下联合决策问题：

```text
给定起点机器狗位姿和目标设备，
从所有安全、可观察的终点位姿中自动选择一个，
同时生成从起点到该终点的低代价路径。
```

算法的输出不是单独的二维路线，而是：

- 机器狗路径 `(x, y, yaw)`；
- 最终停靠位置和机身航向；
- 云台水平角 `pan` 与俯仰角 `tilt`；
- 路径代价、终点观测代价和总代价。

对应实现：

```text
src/substation_vln/planning/common/
src/substation_vln/planning/improved_astar/
tools/planning/build_planning_map.py
tools/planning/build_inspection_goal_regions.py
tools/planning/run_region_goal_astar.py
```

## 2. 输入、输出与坐标约定

### 2.1 输入

算法使用以下输入：

1. 地面已经校正到 `z=0` 附近的完整点云；
2. 正射图与像素—工程坐标变换；
3. 合并后的二维标注；
4. 机器狗尺寸和离散航向数；
5. 相机高度、偏移量、云台角度范围；
6. 不同设备类型的观察高度和推荐俯仰角。

二维标注只要求当前规划实际使用的信息：规划边界、障碍物、优先通过区、优先路径、狭窄空间和设备占地区域。

### 2.2 坐标与状态

工程坐标使用 `(x,y,z)`，地面为 X-Y 平面。规划栅格坐标使用 `(r,c)`。机器狗航向角按 `K` 个方向离散：

\[
\psi_k=\frac{2\pi k}{K},\qquad k=0,1,\ldots,K-1.
\]

当前 `K=16`，角分辨率为 `22.5°`。改进式 A* 状态为：

\[
s=(r,c,k).
\]

### 2.3 输出文件

公共地图输出到：

```text
outputs/<site>/planning/maps/
```

目标位姿区域输出到：

```text
outputs/<site>/planning/inspection_goal_regions/
```

单次路径结果输出到：

```text
outputs/<site>/planning/region_goal_astar/
```

这些文件均可由数据和配置重新生成，不属于人工标注源数据。

## 3. 公共二维规划地图

### 3.1 标注栅格化

`build_planning_map.py` 根据规划边界建立分辨率为 `q` 的规则栅格。当前配置为：

\[
q=0.05\text{ m/cell}.
\]

生成的基础层包括：

- `boundary_mask`：规划边界；
- `obstacle_mask`：障碍物；
- `preferred_road_mask`：优先通过区；
- `preferred_path_mask`：优先路径带；
- `narrow_space_mask`：狭窄空间；
- `equipment_mask`：全部设备占地区域；
- `equipment_index_mask`：每台设备的独立整数索引。

新标注文件按照“一圆或一个多边形一个设备”保存。为兼容旧数据，公共地图层仍会把旧的多圆、多多边形设备标注拆成独立设备索引。

### 3.2 普通 A* 的硬膨胀空间

baseline 将机器人近似为圆，通过障碍物距离变换构造膨胀障碍：

\[
M_{inflated}(p)=\mathbf{1}\left[D_{obs}(p)\le r_{inflate}\right].
\]

当前 `r_inflate=0.4 m`。该空间只服务于普通二维 A* 对照算法。

### 3.3 语义代价图

优先路径吸引场为：

\[
A_{path}(p)=\exp\left(-\frac{D_{path}(p)^2}{2\sigma^2}\right).
\]

障碍物软排斥为：

\[
R_{obs}(p)=w_o
\left[\max\left(\frac{r_o-D_{obs}(p)}{r_o},0\right)\right]^2.
\]

自由栅格的综合语义代价可写为：

\[
C(p)=C_{base/road}(p)-\alpha A_{path}(p)
+P_{narrow}(p)+R_{obs}(p).
\]

其中优先通过区使用较低基础代价，狭窄空间增加惩罚。最终代价不小于 `min_cost`。

改进式 A* 使用 `pose_cost_map`。该图不使用 baseline 的圆形硬膨胀带，因为机器狗矩形足迹将在每个航向下单独检查；但它仍保留障碍物距离软惩罚，使路径主动远离障碍物。

## 4. 独立设备三维几何

### 4.1 一次扫描全部设备

`equipment_index_mask` 将点云的 X-Y 坐标直接映射到设备索引。算法分块扫描大规模 PLY，一次提取全部设备点，避免每台设备分别遍历完整点云。

地面过滤条件为：

\[
z>z_{ground}+h_{clearance}.
\]

当前 `z_ground=0`，`h_clearance=0.15 m`。

### 4.2 稳健包围盒

对设备点云分别计算低、高百分位并去除离群点。当前使用 `1%` 和 `99.5%`。稳健边界记为：

\[
\mathbf b_{min}=(x_{min},y_{min},z_{min}),\qquad
\mathbf b_{max}=(x_{max},y_{max},z_{max}).
\]

设备几何中心为：

\[
\mathbf c=\frac{\mathbf b_{min}+\mathbf b_{max}}{2}.
\]

结果缓存在 `equipment_geometry.json`。设备列表、名称、类型或二维轮廓改变时，缓存签名失效并重新扫描。

## 5. 相机约束下的目标位姿区域

### 5.1 观察高度区间

设设备稳健高度为：

\[
H=z_{max}-z_{min}.
\]

配置观察比例 `[f_l,f_u]` 后：

\[
z_l=z_{min}+f_lH,\qquad
z_u=z_{min}+f_uH,
\]

\[
z_{roi}=\frac{z_l+z_u}{2}.
\]

当前设备配置为：

| 设备 | 观察高度 | 俯仰角范围 | 推荐角度 |
|---|---:|---:|---:|
| 变压器 `bianyaqi` | 0%–100% | 30°–60° | 45° |
| 其他设备 | 50%–100% | 30°–70° | 45° |

变压器观察完整高度；断路器、刀闸、避雷器和互感器主要观察上半部本体，避免支撑架高度把停靠距离推得过远。

### 5.2 圆锥与环形距离区间

相机高度为 `z_cam`。观察中心与相机的高度差为：

\[
\Delta z=z_{roi}-z_{cam}.
\]

在俯仰角 `θ` 下，水平距离为：

\[
d(\theta)=\frac{\Delta z}{\tan\theta}.
\]

由于 `d(θ)` 随角度增大而减小，硬角度范围 `[θ_min,θ_max]` 对应：

\[
d_{min}=\max\left(d_{near},\frac{\Delta z}{\tan\theta_{max}}\right),
\]

\[
d_{max}=\min\left(d_{search},\frac{\Delta z}{\tan\theta_{min}}\right).
\]

因此，相机高度平面与圆锥角范围相交后形成环形候选区域。它不是按设备类别直接规定固定停车距离，而是由设备高度、相机高度和俯仰角共同决定。

### 5.3 机器狗矩形足迹与航向

机器狗尺寸为长 `0.8 m`、宽 `0.4 m`，当前不附加额外几何安全边界。对每个离散航向构造旋转矩形核，并对以下原始自由空间进行腐蚀：

```text
规划边界内
且不属于障碍物
且不属于任何设备占地区域
```

得到：

\[
M_{free}(k,r,c)\in\{0,1\}.
\]

它表示机器狗中心位于 `(r,c)`、航向为 `k` 时完整矩形足迹是否安全。

候选终点的机身方向取设备径向的两个切线方向：

\[
\psi_b=\alpha\pm\frac{\pi}{2},
\]

其中 `α` 是设备中心指向机器狗位置的径向角。该设计使机器狗横向面对设备，同时机身纵轴沿设备外围切线，便于继续沿道路运动。

### 5.4 云台姿态

考虑相机相对机器狗中心的前向、侧向安装偏移后，重新计算相机位置。云台水平角为：

\[
\phi_{pan}=wrap(\alpha_{camera\rightarrow roi}-\psi_b).
\]

俯仰角为：

\[
\theta=\arctan2(\Delta z,d).
\]

候选位姿必须满足云台的水平和俯仰硬限制。当前云台水平范围为 `-180°～180°`，因此主要限制来自设备观察 profile 的俯仰角。

### 5.5 推荐俯仰角软代价

设硬范围为 `[θ_min,θ_max]`，推荐值为 `θ_*`。归一化软代价采用分段线性形式：

\[
J_{tilt}(\theta)=
\begin{cases}
\dfrac{\theta_*-\theta}{\theta_*-\theta_{min}}, & \theta\le\theta_*;\\
\dfrac{\theta-\theta_*}{\theta_{max}-\theta_*}, & \theta>\theta_*.
\end{cases}
\]

推荐角度代价为 `0`，两端硬限制处为 `1`。它不会改变路径中间栅格，只作为终点代价参与最终目标位姿选择。

## 6. 三维点云视线遮挡

### 6.1 场景体素化

完整点云按 `v=0.2 m` 体素化，只保留地面以上 `0.15 m` 到 `35 m` 的占用体素。体素中心建立 `cKDTree`，缓存为 `visibility_voxels.npz`。

### 6.2 加粗中心视线通道

对每个候选相机位置到 `ROI` 中心的线段按 `0.1 m` 采样。若任一采样点附近存在占用体素，该候选位姿被删除。

配置半径为 `r_clear`。考虑体素中心对真实占用范围的误差后，有效检测半径近似为：

\[
r_{eff}=r_{clear}+\frac{\sqrt 3}{2}v.
\]

当前 `r_clear=0.4 m`、`v=0.2 m`，所以 `r_eff≈0.573 m`。这相当于固定半径的圆柱形视线通道，而不是数学上的单条射线。

为避免数值误判，相机端和目标端各忽略 `0.4 m`。计算某台设备时，只排除该设备自身索引下的体素；其他设备、构架和点云障碍仍保留，因此能够剔除被相邻设备遮挡的候选位置。

当前方法只保证 ROI 中心通道无遮挡，不保证整个设备轮廓完全可见。若后续升级为多视线可见率，应作为新的实验变量单独评估。

## 7. 区域目标位姿集合

经过距离、矩形碰撞、云台限制和点云遮挡筛选后，每台设备得到：

\[
\mathcal G=\{(r_i,c_i,k_i,\phi_i,\theta_i,J_{tilt,i})\}.
\]

主要保存字段为：

```text
goal_equipment_index
goal_rows, goal_cols
goal_heading_bins
goal_camera_pan_rad
goal_camera_tilt_rad
goal_tilt_costs
pose_free_packed
```

`pose_free_packed` 使用 bit packing 压缩逐航向碰撞掩码。复核图只显示目标设备和目标位姿区域；颜色饱和度表示终点俯仰代价。

## 8. 改进式 A* 目标函数

### 8.1 状态转移

位姿搜索包含两类动作：

1. 原地向相邻航向 bin 左转或右转；
2. 保持当前机身航向，向 4 邻域或 8 邻域平移。

对角移动还要求两个相邻正交位置均可行，避免穿越障碍物角点。

当前模型允许侧向和后向平移，但根据运动方向与机身方向的偏差增加代价；它不是严格的四足机器人动力学模型。

### 8.2 平移代价

从状态 `s` 平移到 `s'` 的代价为：

\[
g_{move}=lq\left[1+w_cC(p')+w_l(1-|\cos(\beta-\psi)|)\right],
\]

其中：

- `l` 为栅格步长，正交为 1、对角为 `√2`；
- `q` 为地图分辨率；
- `C(p')` 为语义代价；
- `β` 为运动方向；
- `ψ` 为机身航向；
- `w_l` 为横向运动权重。

当运动方向与机身纵轴平行或反平行时，横向惩罚为零；纯侧向运动时惩罚最大。

### 8.3 旋转代价

每旋转一个航向 bin 的代价为：

\[
g_{rot}=w_r.
\]

当前 `w_r=0.35`。

### 8.4 终点代价与联合优化

对终点状态 `s_g`：

\[
J_{terminal}(s_g)=w_tJ_{tilt}(s_g).
\]

当前 `w_t=15`。最终优化目标为：

\[
s_g^*=\arg\min_{s_g\in\mathcal G}
\left[J_{path}(s_0\rightarrow s_g)+J_{terminal}(s_g)\right].
\]

因此算法不会默认选择几何距离最近或最远的停靠点，而是在运动成本、道路语义、障碍物距离、机身姿态和相机推荐角度之间联合权衡。

## 9. 两阶段分层搜索

完整状态空间大小为 `K×H×W`。在当前高分辨率地图上直接全局搜索会占用大量时间和内存，因此实现采用粗到细策略。

### 9.1 第一阶段：二维区域目标 A*

先将逐航向可行掩码合并为位置可行性：

\[
M_{position}(r,c)=\max_k M_{free}(k,r,c).
\]

同一位置若有多个目标航向，取最小终点代价。第一阶段在二维位置空间中搜索到任意目标位置，并通过目标区域距离变换构造启发式：

\[
h(p)=D_{goal}(p)\,q\,[1+w_cC_{min}].
\]

搜索不会在找到第一个目标位置时立即返回，而是比较路径代价与该位置的终点代价，选择当前总成本最低的目标。

### 9.2 路径走廊

将二维粗路径按半径 `R` 膨胀为走廊：

\[
M_{corridor}=dilate(path,R/q).
\]

当前初始半径为 `2 m`。若位姿搜索失败，半径依次扩大为 `4 m`、`8 m`。

### 9.3 第二阶段：走廊内位姿 A*

在 `M_corridor` 内对 `(r,c,k)` 搜索，完整检查：

- 当前航向的矩形足迹；
- 平移语义代价和横向运动代价；
- 原地旋转代价；
- 终点位置与机身航向；
- 终点相机软代价。

启发式仍使用到目标位置集合的距离变换，不把终点软代价加入启发式。找到候选目标后，只有当开放队列的下界已经不可能优于当前最佳总成本时才结束。

该方法显著减少位姿状态展开数，但搜索范围受最大走廊半径限制，因此是面向工程效率的分层实现，不应表述为无条件的全局完整性保证。

## 10. 配置与代码对应关系

### 10.1 地图配置

```text
configs/tools/planning/build_planning_map.yaml
```

控制栅格分辨率、障碍膨胀、障碍物排斥、道路代价、优先路径吸引和狭窄空间惩罚。

### 10.2 目标区域配置

```text
configs/tools/planning/build_inspection_goal_regions.yaml
```

控制机器狗尺寸、航向数、相机参数、设备观察 profile、点云几何提取、视线体素和候选采样步长。

### 10.3 搜索配置

```text
configs/tools/planning/run_region_goal_astar.yaml
```

控制初始航向、走廊半径、路径代价权重、旋转代价、横向移动代价和终点俯仰代价。

目标区域复核图与搜索必须使用相同的 `tilt_cost_weight`，否则颜色表达的偏好与实际终点选择不一致。

## 11. 标准运行与重建规则

### 11.1 完整流程

```bash
python substation_vln/tools/annotation/merge_annotation_files.py
python substation_vln/tools/planning/build_planning_map.py
python substation_vln/tools/planning/build_inspection_goal_regions.py
python substation_vln/tools/planning/run_region_goal_astar.py
```

### 11.2 非交互测试

```bash
python substation_vln/tools/planning/run_region_goal_astar.py \
  --equipment <设备索引或完整名称> \
  --start-x <工程坐标X> \
  --start-y <工程坐标Y> \
  --start-yaw-deg 0 \
  --no-display
```

### 11.3 何时重建缓存

- 仅修改 A* 搜索权重：直接重新运行 `run_region_goal_astar.py`；
- 修改俯仰角、采样步长或视线通道参数：重新运行 `build_inspection_goal_regions.py`；
- 修改设备标注：重新 merge、构建地图，并增加 `--rebuild-geometry`；
- 修改完整点云或体素参数：增加 `--rebuild-visibility`。

## 12. 验证指标

与普通固定终点 A* 对比时，建议至少记录：

- 是否找到可行路径；
- 路径长度和总代价；
- 障碍物最小距离；
- 展开节点数和运行时间；
- 最终停靠距离、机身航向和云台角度；
- 候选位姿总数、遮挡剔除比例；
- 到达设备邻域后局部 VLN 的运动步数和拍摄成功率。

消融实验可分别移除目标区域、机身航向、障碍物软排斥、推荐俯仰代价、点云遮挡和分层走廊，以说明各设计对安全性、效率和拍摄初始条件的贡献。
