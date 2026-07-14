# 三维点云可行巡视区域计算技术方案

## 1. 目标

本模块根据三维巡视目标、完整点云和二维安全地图，自动生成机器人可以安全停靠且目标具有鲁棒无遮挡视线的二维区域。

```text
三维目标点
→ 三维观测距离筛选
→ 有半径的鲁棒视线通道检测
→ 二维硬安全空间求交
→ 可行巡视区域
```

核心代码：

```text
substation_vln/src/substation_vln/inspection_regions/voxel_map.py
substation_vln/src/substation_vln/inspection_regions/visibility_corridor.py
substation_vln/src/substation_vln/inspection_regions/feasible_region.py
substation_vln/src/substation_vln/inspection_regions/io.py
```

命令行入口与配置：

```text
substation_vln/tools/planning/build_feasible_inspection_regions.py
substation_vln/configs/tools/planning/build_feasible_inspection_regions.yaml
```

配置与脚本使用完全相同的基名 `build_feasible_inspection_regions`。

三维巡视目标从统一的 `outputs/220kv_erfeishan/annotation/annotations_merged.json` 中筛选，条件为 `category=inspection_target` 且 `geometry_type=point_3d`；不再读取独立目标文件。`equipment_id` 用于将同一物理设备的多个巡视点位分组，当前脚本仍逐点生成可行域，后续在这些可行域上求解多目标覆盖与最少停靠点。

## 2. 坐标与相机假设

点云预处理已经将拟合基准地面设置为：

\[
z_{ground}=0.
\]

机器人候选停靠位置为二维栅格中心：

\[
g=(x,y,0).
\]

相机假设为离地1米的全向相机：

\[
c(g)=(x,y,1).
\]

三维巡视目标为设备表面点：

\[
t_i=(x_i,y_i,z_i).
\]

当前离线阶段不计算相机偏航、水平视场角、图像清晰度和目标居中程度。

## 3. 稀疏点云体素化

设体素尺寸为 \(s_v\)，点云原点为 \(o\)，点 \(p\) 的体素索引为：

\[
k(p)=\left\lfloor\frac{p-o}{s_v}\right\rfloor.
\]

当前使用：

```yaml
voxel_size_m: 0.10
```

实现采用流式内存映射读取，每次处理 `chunk_size` 个点；每个数据块先局部去重，最后合并为全局稀疏占据体素索引。不会创建覆盖整个变电站的稠密三维数组。

体素结果缓存到：

```text
outputs/220kv_erfeishan/planning/inspection_regions/cache/pointcloud_voxels_0p10m.npz
```

缓存同时记录源点云路径、修改时间和体素尺寸。三者一致时自动复用；点云或体素参数改变时自动重建。

## 4. 距离候选区域

相机到目标的三维距离为：

\[
d_i(g)=\|c(g)-t_i\|_2.
\]

距离有效条件：

\[
d_{min,i}\le d_i(g)\le d_{max,i}.
\]

设目标与相机高度差为：

\[
\Delta z=z_i-1,
\]

则水平最大搜索半径为：

\[
r_{xy,max}=\sqrt{d_{max,i}^2-\Delta z^2}.
\]

算法只遍历目标投影附近的包围盒，不扫描整张二维地图。

## 5. 鲁棒视线通道

### 5.1 视线半径

配置：

```yaml
visibility_clearance_radius_m: 0.2
```

即：

\[
r_{los}=0.2\text{ m}.
\]

它表示相机—目标中心线周围半径0.2米、直径0.4米的视线通道必须没有障碍物。

### 5.2 方法1：膨胀障碍后检测中心线

对目标附近的局部占据体素块使用欧氏球形结构元素膨胀：

\[
V_{inflated}=V\oplus B(r_{los}).
\]

再检测中心线是否与膨胀体素相交：

\[
L(c,t_i)\cap V_{inflated}=\varnothing.
\]

该操作近似等价于检测半径为 \(r_{los}\) 的圆柱或胶囊形视线通道是否与原始障碍相交。

当 `visibility_clearance_radius_m=0` 时不膨胀，退化为单射线基线。

### 5.3 端点排除

原始射线不能从相机中心一直检查到目标点，否则目标设备自身会被当成遮挡。膨胀后使用有效排除距离：

\[
r_{camera}^{eff}=r_{camera}+r_{los},
\]

\[
r_{target}^{eff}=r_{target}+r_{los}.
\]

当前参数为：

```text
camera_exclusion_radius_m = 0.3 m
target_exclusion_radius_m = 0.1 m
visibility_clearance_radius_m = 0.2 m

有效相机端排除 = 0.5 m
有效目标端排除 = 0.3 m
```

实际检测线段为：

\[
L=
\left[
c+r_{camera}^{eff}\hat v,
t_i-r_{target}^{eff}\hat v
\right].
\]

## 6. 加速策略

### 6.1 全局稀疏、局部稠密

全站点云保存为稀疏占据索引。对每个目标只裁剪：

```text
目标XY周围最大观测距离
相机高度到目标高度
视线半径和体素边界余量
```

得到局部稠密布尔数组，然后只对这个小数组执行膨胀和射线检测。

### 6.2 Numba并行3D DDA

候选相机位置组成批量数组。每条射线采用Amanatides-Woo风格的3D DDA遍历体素，并通过Numba `parallel=True`并行计算。

当前后端记录为：

```text
numba_parallel_3d_dda
```

### 6.3 缓存复用

第一次处理6754万点约需数秒建立缓存；后续目标无需重新体素化，只需加载缓存和处理局部体素块。

## 7. 二维安全求交

距离有效且鲁棒视线无遮挡：

\[
M_i^{visible}
=M_i^{distance}\cap M_i^{visibility}.
\]

二维硬安全空间为：

\[
M^{free}=M^{boundary}\cap\neg M^{inflated\_obstacle}.
\]

最终可行巡视区域：

\[
M_i^{feasible}=M_i^{visible}\cap M^{free}.
\]

狭窄空间仍属于自由空间，因此可以成为停靠区域；它只在后续路径规划中产生额外软代价。

## 8. 区域清理

最终mask可执行：

```yaml
min_region_area_m2: 0.5
morphology_open_radius_m: 0.0
```

当前删除面积小于0.5平方米的连通分量，不进行形态学开运算。建议保留 `feasible_inspection_region_raw_mask` 与清理后的 `feasible_inspection_region_mask`，方便论文消融。

## 9. 运行命令

处理全部目标：

```bash
conda run -n habitat-gs python \
  substation_vln/tools/planning/build_feasible_inspection_regions.py
```

只处理指定目标：

```bash
conda run -n habitat-gs python \
  substation_vln/tools/planning/build_feasible_inspection_regions.py \
  --target-id target_001
```

强制重建体素缓存：

```bash
conda run -n habitat-gs python \
  substation_vln/tools/planning/build_feasible_inspection_regions.py \
  --rebuild-voxel-cache
```

## 10. 输出

每个目标输出到：

```text
outputs/220kv_erfeishan/planning/inspection_regions/<target_id>/
```

包含：

```text
feasible_inspection_region.npz
distance_candidate_mask.png
robust_visibility_mask.png
visible_inspection_region_mask.png
feasible_inspection_region_raw_mask.png
feasible_inspection_region_mask.png
distance_to_target.png
feasible_inspection_region_overlay.png
metadata.json
```

叠加图颜色：

```text
红色十字：三维目标XY投影
黄色：三维距离与视线有效，但二维位置不一定安全
绿色：最终可行巡视区域
灰色：二维不可通行区域
```

## 11. 当前真实数据结果

对 `target_001`，当前参数得到：

```text
点云点数：67,548,769
0.1 m占据体素：7,421,654
距离有效栅格：471,220
鲁棒可见栅格：19,184
二维安全求交后：17,341
区域清理后：17,284
可行巡视连通区域：3
单目标计算时间：约0.6 s（不含首次体素缓存）
```

首次体素缓存约6.2秒，后续运行自动复用。

## 12. 参数敏感性实验

建议重点比较：

```text
voxel_size_m: 0.10, 0.15, 0.20
visibility_clearance_radius_m: 0.0, 0.1, 0.2, 0.3
target_exclusion_radius_m: 0.1, 0.2, 0.3
min_region_area_m2: 0.0, 0.5, 1.0
```

评价指标：

```text
可行巡视区域面积
连通分量数量
人工复核可见率
假可见率和假遮挡率
单目标计算时间
后续区域目标A*规划成功率与路径长度
```
