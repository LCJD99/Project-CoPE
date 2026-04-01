# Data Policy

Commit only small metadata/manifests/samples.  
Do not commit large raw datasets, model weights, caches, or generated run outputs.

## Directory Contract

- `data/00_global`: shared static inputs (system config, registry, profiling)
- `data/01_stage1`: Stage1 training data
- `data/02_stage2`: Stage2 training data
- `data/03_stage3`: Stage3 RL data
