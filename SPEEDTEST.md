# Karpathy Nanochat

## Speed Test - Vanilla 788110ad

- 1x RTX 3090, pl=350W, bs=4

```bash
sudo nvidia-smi -pl 350
CUDA_VISIBLE_DEVICES=1 python -m scripts.base_train --depth=12 --device_batch_size=4
step 00011/02832 (0.39%) | loss: 7.285519 | lrm: 1.00 | dt: 8861.59ms | tok/sec: 59,164 | mfu: 6.21 | total time: 0.15m | eta: 416.6m
```

- 2x RTX 3090, pl=350W, bs=4

```bash
sudo nvidia-smi -pl 350
CUDA_VISIBLE_DEVICES=1,3 torchrun --standalone --nproc_per_node=2 -m  scripts.base_train --depth=12 --device_batch_size=4
step 00011/02832 (0.39%) | loss: 7.292456 | lrm: 1.00 | dt: 4686.39ms | tok/sec: 111,874 | mfu: 5.87 | total time: 0.08m | eta: 220.3m
```

- 2x RTX 3090, pl=250W, bs=4

```bash
sudo nvidia-smi -pl 250
CUDA_VISIBLE_DEVICES=1,3 torchrun --standalone --nproc_per_node=2 -m  scripts.base_train --depth=12 --device_batch_size=4
step 00011/02832 (0.39%) | loss: 7.292482 | lrm: 1.00 | dt: 5230.18ms | tok/sec: 100,242 | mfu: 5.26 | total time: 0.09m | eta: 245.9m
```

- 4x RTX 3090, pl=250W, bs=4

```bash
sudo nvidia-smi -pl 250
torchrun --standalone --nproc_per_node=4 -m  scripts.base_train --depth=12 --device_batch_size=4
step 00011/02832 (0.39%) | loss: 7.218103 | lrm: 1.00 | dt: 3030.89ms | tok/sec: 172,981 | mfu: 4.54 | total time: 0.05m | eta: 142.5m
```

- 4x RTX 3090, pl=250W, bs=8

```bash
sudo nvidia-smi -pl 250
torchrun --standalone --nproc_per_node=4 -m  scripts.base_train --depth=12 --device_batch_size=8
step 00011/02832 (0.39%) | loss: 7.284660 | lrm: 1.00 | dt: 2907.47ms | tok/sec: 180,324 | mfu: 4.73 | total time: 0.05m | eta: 136.7m
```

- 4x RTX 3090, pl=250W, bs=16

```bash
# likely air cooled gpu is temp throttling
sudo nvidia-smi -pl 250
torchrun --standalone --nproc_per_node=4 -m  scripts.base_train --depth=12 --device_batch_size=16
step 00011/02832 (0.39%) | loss: 7.328738 | lrm: 1.00 | dt: 2941.21ms | tok/sec: 178,255 | mfu: 4.68 | total time: 0.05m | eta: 138.3m
```

- 4x RTX 3090, pl=200W, bs=16

```bash
sudo nvidia-smi -pl 200
torchrun --standalone --nproc_per_node=4 -m  scripts.base_train -- --depth=12 --device_batch_size=16 --run=d12
step 00011/02832 (0.39%) | loss: 7.328709 | lrm: 1.00 | dt: 3454.68ms | tok/sec: 151,761 | mfu: 3.98 | total time: 0.06m | eta: 162.4m
```


## Speed Test - Modified Deterministic 788110ad

- 1x RTX 3090 watercooled, power limit 350W
- torch.compile commented out
- added before model creation:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

```bash
sudo nvidia-smi -pl 350
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m scripts.base_train --depth=12 --device_batch_size=4
step 00011/02832 (0.39%) | loss: 7.285553 | lrm: 1.00 | dt: 16501.75ms | tok/sec: 31,771 | mfu: 3.33 | total time: 0.28m | eta: 775.9m
```


## Models and Sizes -  Vanilla 788110ad - PL 200W

```bash

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --depth=12 --device_batch_size=16
step 00011/02832 (0.39%) | loss: 7.328726 | lrm: 1.00 | dt: 3460.44ms | tok/sec: 151,509 | mfu: 3.98 | total time: 0.06m | eta: 162.7m


torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --depth=10 --device_batch_size=16
step 00011/02030 (0.54%) | loss: 7.350708 | lrm: 1.00 | dt: 2489.80ms | tok/sec: 210,574 | mfu: 3.75 | total time: 0.04m | eta: 83.8m


torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --depth=8 --device_batch_size=16
step 00011/01408 (0.78%) | loss: 7.394463 | lrm: 1.00 | dt: 1613.64ms | tok/sec: 324,910 | mfu: 51.82 | total time: 0.03m | eta: 37.6m

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- --depth=6 --device_batch_size=32
step 00011/00930 (1.18%) | loss: 7.447861 | lrm: 1.00 | dt: 1044.52ms | tok/sec: 501,943 | mfu: 47.95 | total time: 0.02m | eta: 16.0m
```