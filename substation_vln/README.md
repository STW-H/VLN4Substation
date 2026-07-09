# Substation VLN Workspace

This workspace organizes data and tools for the 220 kV Erfeishan substation
inspection navigation project.

## Layout

```text
substation_vln/
├── configs/                         # Experiment configs
├── data/
│   ├── raw/220kv_erfeishan/          # Original data as symlinks
│   │   ├── pointcloud/               # Raw LiData point cloud
│   │   ├── gaussian/                 # Raw 3DGS PLY files
│   │   ├── metadata/                 # GPS and other metadata
│   │   └── viewer/                   # Original Windows GS viewer
│   └── processed/220kv_erfeishan/
│       ├── pointcloud/               # Converted PLY/PCD/LAS point clouds
│       ├── gaussian/                 # Habitat-GS friendly *.gs.ply links
│       ├── navmesh/                  # Future NavMesh files
│       ├── maps/                     # Future 2D maps
│       ├── semantic/                 # Future semantic maps
│       └── safety/                   # Future safety constraint maps
├── outputs/220kv_erfeishan/          # Rendered images, videos, figures
├── src/substation_vln/               # Future Python package
└── tools/                            # Standalone tools
```

The raw data is linked or stored under `data/raw`. The converted LAS point cloud
has also been exported to a real-world-coordinate binary PLY under
`data/processed/220kv_erfeishan/pointcloud/`.

## Quick Commands

View the processed real-coordinate point cloud. The viewer loads all points by
default and centers the display for easier inspection:

```bash
conda activate habitat-gs
python substation_vln/tools/view_pointcloud.py \
  substation_vln/data/processed/220kv_erfeishan/pointcloud/erfeishan_0.02_resampled_real_coords.ply
```

Regenerate the real-coordinate PLY from LAS:

```bash
conda activate habitat-gs
python substation_vln/tools/view_pointcloud.py \
  substation_vln/data/raw/220kv_erfeishan/pointcloud/erfeishan_0.02_resampled.las \
  --save-converted \
  --info
```

View a Gaussian layer with Habitat-GS:

```bash
conda activate habitat-gs
python substation_vln/tools/view_gaussian.py \
  substation_vln/data/processed/220kv_erfeishan/gaussian/layer_0.gs.ply
```

Render a quick offscreen preview:

```bash
conda activate habitat-gs
python substation_vln/tools/view_gaussian.py \
  substation_vln/data/processed/220kv_erfeishan/gaussian/layer_0.gs.ply \
  --snapshot --output substation_vln/outputs/220kv_erfeishan/gaussian/layer_0_preview.png
```
