# VLN4Substation

变电站巡检机器人视觉语言导航研究原型。当前项目围绕 220 kV 二妃山变电站数据，完成点云/3D Gaussian 数据组织、可视化、坐标对齐，并为后续二维地图、可通行区域标注和局部主动观测位姿优化提供基础工具链。

## Main Functionality

当前仓库主要包含：

```text
1. LAS/LAZ 点云转换为真实坐标 PLY
2. 完整点云可视化与坐标轴查看
3. Habitat-GS 3D Gaussian 渲染查看，脚本自动维护 outputs 下的 Z-up 到 Habitat Y-up 可视化缓存
4. LAS 点云坐标轴矫正
5. Gaussian 到完整点云的手工点粗配准 + ICP 精配准
6. 对齐后的 Gaussian 点云预处理输出
7. ICP 滤波效果预览
8. 正射图交互标注与多次标注合并
9. 规划边界、障碍物、道路和有向/无向路径地图构建
10. 交互式 baseline A* 链路测试
11. 相机成像约束下的巡视目标位姿区域生成
12. 矩形机器狗区域目标位姿 A*
```

当前算法思路见：

```text
substation_vln/docs/algorithm_overview.md
```

## Repository Layout

```text
.
├── environment.yml                # 本项目工具依赖参考
├── external/                      # Habitat-GS / Habitat-Lab 等外部依赖，本地准备，不提交 Git
├── substation_vln/
│   ├── src/substation_vln/         # 可复用 Python 模块，按功能域组织
│   │   ├── preprocessing/          # 点云/高斯/坐标预处理基础函数
│   │   ├── annotation/             # 标注数据结构与交互标注器
│   │   ├── visualization/          # 点云与高斯可视化基础函数
│   │   └── planning/               # 路径规划算法，含 A* 与改进 A*
│   ├── tools/                      # 命令行工具入口，按流程模块分组
│   │   ├── preprocessing/          # 点云/高斯预处理与配准
│   │   ├── annotation/             # 标注与标注合并
│   │   ├── visualization/          # 点云和高斯可视化
│   │   └── planning/               # 规划地图构建与 A* 测试入口
│   ├── docs/                       # 算法思路文档
│   ├── configs/                    # 后续实验配置
│   ├── data/                       # 本地数据目录，不提交 Git
│   └── outputs/                    # 可视化和实验输出；当前 planning 结果允许提交
├── 变电站巡检机器人视觉语言导航研究方案.md
└── .gitignore
```

## Environment Setup

建议系统：

```text
OS: Ubuntu 24.04
GPU: NVIDIA GPU
Conda: Miniconda / Anaconda
```

先检查 GPU 和 OpenGL：

```bash
nvidia-smi
glxinfo | grep "OpenGL version"
```

如果没有 `glxinfo`：

```bash
sudo apt update
sudo apt install mesa-utils
```

### 1. Clone Repository

```bash
git clone https://github.com/STW-H/VLN4Substation.git
cd VLN4Substation
```

### 2. Prepare External Dependencies First

本仓库默认不提交 `external/`。请先准备：

```text
external/
├── habitat-gs/
└── habitat-lab/
```

建议顺序：

```text
1. 安装 Habitat-GS
2. 安装或准备 Habitat-Lab
3. 确认 Habitat-GS 中集成的 Habitat-Sim 可用
4. 再安装本仓库工具所需的额外 Python 包
```

Habitat-GS / Habitat-Sim 对 CUDA、显卡驱动、编译器和 Ubuntu 版本比较敏感，应优先按照 Habitat-GS 官方 README 安装。当前项目不把 external 作为 submodule 管理；如果需要严格复现，建议 fork Habitat-GS 并记录 commit 与本地补丁。

当前本地使用中涉及过的 Habitat-GS 兼容性处理包括：

```text
CUDA 架构适配 RTX 4060 Laptop GPU
Ubuntu 24.04 / 新 CUDA 版本兼容处理
matplotlib 新版本 colormap API 兼容处理
gaussian_viewer.py 增加启动位置和 yaw 参数
```

### 3. Conda Environment

如果已经按照 Habitat-GS 安装好了环境，直接激活：

```bash
conda activate habitat-gs
```

然后补充本项目工具依赖：

```bash
pip install open3d laspy lazrs pillow matplotlib scipy
```

也可以使用仓库中的 `environment.yml` 作为参考创建环境：

```bash
conda env create -f environment.yml
conda activate habitat-gs
```

注意：`environment.yml` 只覆盖本仓库工具层常用依赖，不替代 Habitat-GS / Habitat-Sim 的源码编译安装。更稳妥的方式通常是：先按 Habitat-GS 官方流程建好 `habitat-gs` 环境，再用上面的 `pip install` 补齐本项目工具包。

## Data Placement

本仓库不包含点云和高斯大文件。建议本地组织为：

```text
substation_vln/data/raw/220kv_erfeishan/
├── gaussian/
├── pointcloud/
├── metadata/
└── viewer/

substation_vln/data/processed/220kv_erfeishan/
├── gaussian/
├── pointcloud/
├── registration/
├── maps/
├── navmesh/
├── safety/
└── semantic/
```

当前常用文件：

```text
substation_vln/data/raw/220kv_erfeishan/gaussian/layer_2_point_cloud.ply
substation_vln/data/processed/220kv_erfeishan/pointcloud/erfeishan_0.02_resampled_real_coords_axis_corrected.ply
substation_vln/data/processed/220kv_erfeishan/gaussian/layer_2_aligned_to_axis_corrected_pointcloud.ply
substation_vln/data/processed/220kv_erfeishan/registration/gaussian_to_pointcloud_transform.json
substation_vln/outputs/220kv_erfeishan/annotation/annotations_merged.json
```

## Common Commands

查看完整点云：

```bash
python substation_vln/tools/visualization/view_pointcloud.py \
  substation_vln/data/processed/220kv_erfeishan/pointcloud/erfeishan_0.02_resampled_real_coords.ply
```

LAS 转真实坐标 PLY，并可选做坐标轴矫正：

```bash
python substation_vln/tools/preprocessing/convert_las_to_real_ply.py \
  substation_vln/data/raw/220kv_erfeishan/pointcloud/erfeishan_0.02_resampled.las \
  --axis-correct
```

查看 3D Gaussian 渲染：

```bash
python substation_vln/tools/visualization/view_gaussian.py
```

高斯点云注册到完整点云：

```bash
python substation_vln/tools/preprocessing/register_gaussian_to_pointcloud.py \
  --correspondences substation_vln/data/processed/220kv_erfeishan/registration/gaussian_to_pointcloud_transform.json
```

重新手工选点配准：

```bash
python substation_vln/tools/preprocessing/register_gaussian_to_pointcloud.py \
  --num-points 6 \
  --pick-order gaussian-first
```

预览 ICP 滤波效果：

```bash
python substation_vln/tools/preprocessing/register_gaussian_to_pointcloud.py \
  --preview-icp-filter substation_vln/data/processed/220kv_erfeishan/registration/gaussian_to_pointcloud_transform.json
```

合并多次二维标注结果：

```bash
python substation_vln/tools/annotation/merge_annotation_files.py
```

从合并标注构建规划地图：

```bash
python substation_vln/tools/planning/build_planning_map.py \
  --config substation_vln/configs/tools/planning/build_planning_map.yaml
```

交互式测试 baseline A*：

```bash
python substation_vln/tools/planning/run_baseline_astar.py
```

生成设备的相机可行目标位姿区域：

```bash
python substation_vln/tools/planning/build_inspection_goal_regions.py
```

运行区域目标位姿 A*：

```bash
python substation_vln/tools/planning/run_region_goal_astar.py
```

标注菜单中的“机器人起始点”支持一次点击多个点。重新合并并构建地图后，可以指定或随机选择起点：

```bash
python substation_vln/tools/planning/run_region_goal_astar.py \
  --equipment 1#duanluqi_1 --start-point gate_1

python substation_vln/tools/planning/run_region_goal_astar.py \
  --equipment 1#duanluqi_1 --random-start --random-seed 42
```

使用 DeepSeek V4 Pro 从自然语言解析起点、途经点、目标设备和运动模式，并生成、保存和展示完整路径：

```bash
# 默认从被Git忽略的 .secrets/deepseek_api_key 读取；也可临时使用：
export DEEPSEEK_API_KEY='your-api-key'
python substation_vln/tools/planning/run_natural_language_route.py
```

输入文字配置在 `configs/tools/planning/run_natural_language_route.yaml` 的 `instruction` 字段。

也可以复用已经解析的任务，避免再次调用 API：

```bash
python substation_vln/tools/planning/run_natural_language_route.py \
  --plan-json substation_vln/outputs/220kv_erfeishan/tasks/inspection_plan_<timestamp>.json
```

三套运动模式参数位于 `configs/tools/planning/modes/`：`normal.yaml`、`fast.yaml` 和 `safe.yaml`。`build_planning_map.py` 根据它们离线生成三套 cost 地图；运行时直接按解析模式读取。硬碰撞约束在三种模式下保持一致。

每个 `tools` 脚本都有对应的默认 YAML 配置，统一放在：

```text
substation_vln/configs/tools/
```

规划中间结果默认输出到：

```text
substation_vln/outputs/220kv_erfeishan/planning/maps/
```

baseline A* 测试结果默认输出到：

```text
substation_vln/outputs/220kv_erfeishan/planning/baseline_astar/
```

`outputs/` 是可重新生成的运行结果，默认不提交 Git。

## Git Policy

点云、高斯文件、渲染输出和外部依赖体积较大，默认不提交到 Git。建议 `.gitignore` 保留：

```gitignore
external/
substation_vln/data/raw/
substation_vln/data/processed/
substation_vln/outputs/
*.ply
*.las
*.laz
*.pcd
*.LiData
__pycache__/
*.pyc
```

仓库主要提交：

```text
README.md
environment.yml
substation_vln/docs/
substation_vln/src/
substation_vln/tools/
substation_vln/configs/
```
