# Novel View Synthesis with Diffusion Models

PyTorch implementation of novel view synthesis with diffusion models.

This repository supports two clean setup paths:

- Conda: local reproducible environment from `environment.yml`
- Docker: CUDA-ready image from `Dockerfile` and `docker-compose.yml`

## Repository Layout

```text
NVSynthesis/                  Core training and testing code
configs/                      YAML experiment configuration
data/                         Local datasets, ignored by Git
experiments/                  Checkpoints and training logs, ignored by Git
results/                      Test outputs, ignored by Git
environment.yml               Conda environment
Dockerfile                    Docker image definition
docker-compose.yml            Docker runtime configuration
requirements.txt              Python app dependencies used by Docker
train.sh / test.sh            Root launchers
```

## Data And Checkpoints

The default config expects MSTAR data at:

```text
data/MSTAR_PUBLIC_MIXED_TARGETS
```

The default test config expects a checkpoint at:

```text
experiments/diffusion_ori/pretrained/models/epoch_1360_iter_690000.pt
```

Update the paths in `configs/_base_/datasets/datasource/mstar/trainset.yml` and
`configs/_base_/NVSynthesis/test.yml` if your files live elsewhere.

## Option 1: Conda

```bash
conda env create -f environment.yml
conda activate nvsynthesis
./train.sh
```

Run testing after placing a checkpoint at the configured path:

```bash
./test.sh
```

Use a different config without editing the launcher:

```bash
NVSYNTHESIS_TRAIN_OPT=/path/to/train.yml ./train.sh
NVSYNTHESIS_TEST_OPT=/path/to/test.yml ./test.sh
```

## Option 2: Docker

Build the image:

```bash
docker compose build
```

Train:

```bash
docker compose run --rm nvsynthesis ./train.sh
```

Test:

```bash
docker compose run --rm nvsynthesis ./test.sh
```

Open an interactive shell:

```bash
docker compose run --rm nvsynthesis bash
```

The compose file mounts `./data`, `./experiments`, and `./results` into the
container, so datasets and generated checkpoints remain on the host.

## GPU Selection

The default training config uses GPU `0`:

```yaml
gpu_ids: [0]
```

For multi-GPU training, edit `configs/_base_/NVSynthesis/train.yml`, for example:

```yaml
gpu_ids: [0, 1, 2, 3]
```

## Direct Commands

The launchers are thin wrappers around:

```bash
python NVSynthesis/apis/main.py --mode train -opt configs/_base_/NVSynthesis/train.yml
python NVSynthesis/apis/main.py --mode test -opt configs/_base_/NVSynthesis/test.yml
```

## License

This project is provided for research and educational purposes. See `LICENSE`.
