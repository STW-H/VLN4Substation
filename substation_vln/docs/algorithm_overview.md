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
├── gaussian_yup/
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
substation_vln/tools/convert_las_to_real_ply.py
substation_vln/tools/view_pointcloud.py --save-converted
```

转换后需要检查：

```text
点数是否正确
XYZ 范围是否合理
RGB 是否正常保留
坐标是否处于真实工程坐标系
```

### 2.3 高斯点云预处理

原始 3D Gaussian 文件通常不能直接用于 Habitat-GS 或当前配准流程，需要先确认坐标轴方向。当前二妃山数据中，原始高斯为 Z-up，Habitat-GS 使用时需要转换为 Y-up。

处理流程：

```text
原始 Gaussian PLY
  -> 使用 Habitat-GS 官方 rotate_gs.py 做坐标轴旋转
  -> 生成 processed/gaussian_yup/*.gs.ply
  -> 使用 Habitat-GS 查看渲染效果
  -> 使用 Open3D 以高斯中心点形式检查结构与颜色
```

当前工具：

```text
substation_vln/tools/rotate_erfeishan_gaussians_yup.sh
substation_vln/tools/view_gaussian.py
```

### 2.4 粗尺度检查

正式配准前，先粗略检查完整点云和原始高斯点云是否存在明显尺度差异或方向差异。

当前工具：

```text
substation_vln/tools/overlay_las_gaussian_raw.py
substation_vln/tools/measure_gaussian_pointcloud_scale.py
```

重点检查：

```text
整体 extent 是否同量级
长边方向是否一致
高度方向是否一致
选取同一物理线段时，两边长度比例是否接近 1
```

如果尺度差异很小，可以继续使用相似变换 + ICP 完成统一坐标对齐。如果尺度差异很大，需要先确认数据来源、坐标单位和高斯训练过程是否引入了尺度归一化。

### 2.5 高斯到完整点云配准

当前以完整点云真实坐标系作为参考坐标系，将 Y-up 高斯点云注册到完整点云。

配准流程：

```text
1. 在高斯点云中选取若干清晰结构点
2. 在完整点云中按相同顺序选择对应结构点
3. 使用 Umeyama 相似变换完成粗配准
4. 对参与 ICP 的高斯点云进行范围、高度和统计离群点过滤
5. 对完整点云和高斯点云执行 ICP 精配准
6. 保存 Gaussian -> PointCloud 的最终变换矩阵
7. 在同一窗口中可视化完整点云和变换后的高斯点云，人工确认效果
```

当前工具：

```text
substation_vln/tools/register_gaussian_to_pointcloud.py
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

### 2.6 预处理结果

完成预处理后，应得到以下基础数据：

```text
完整点云真实坐标 PLY
Y-up 高斯场景文件
Gaussian -> PointCloud 坐标变换矩阵
点云和高斯的可视化检查结果
```

这些数据为后续地图生成、语义标注、安全约束构建和局部观测位姿优化提供统一坐标基础。

## 3. 预处理后的标注流程

完成坐标对齐后，下一阶段进入标注流程。当前阶段只定义标注目标和基本流程，具体标注工具和数据格式后续再完善。

### 3.1 可通行区域标注

首先需要在完整点云或由点云生成的二维投影图上标注机器人可通行区域。

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

## 4. 后续工作

后续主要内容将围绕以下方向继续完善：

```text
二维地图生成
可通行区域与安全约束标注工具
候选巡视点和观测位姿数据结构
局部主动观测策略
VLM/VLN 黑盒评价与局部决策接口
仿真验证流程
```

当前阶段先以“完成可靠预处理和坐标统一”为主，确保后续地图、标注和局部观测算法都建立在同一坐标基础上。

