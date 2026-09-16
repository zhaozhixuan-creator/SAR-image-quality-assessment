# Environment Setup

This project is pinned to the legacy OpenMMLab v1 generation stack used by the
local `mmgen` code: Python 3.8, PyTorch 1.10.2, CUDA 11.3, and `mmcv-full`
1.6.2. The entry points are `train.sh` and `test.sh`.

## Path A: Docker-first

Build the image:

```bash
docker build -t jk-frequency:openmmlab .
```

Run training with GPU access and mounted data/checkpoints:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/work_dirs:/workspace/work_dirs" \
  jk-frequency:openmmlab bash train.sh
```

Run evaluation:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/work_dirs:/workspace/work_dirs" \
  jk-frequency:openmmlab bash test.sh
```

The image intentionally does not bake `data/`, `work_dirs/`, or checkpoint files
into the build context.

## Path B: Conda-first

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate openmmlab
```

Run the project from the repository root:

```bash
bash train.sh
bash test.sh
```

For a different CUDA/PyTorch pair, update both the Conda `pytorch`,
`torchvision`, `cudatoolkit` pins and the `mmcv-full` wheel URL in
`environment.yml` so they remain ABI-compatible.
