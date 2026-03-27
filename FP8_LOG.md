

## Baseline bf16

```bash
MY-NANOCHAT FA3=False FP8=False
python -m scripts.base_train --depth=12 --device-batch-size=8 --window-patter=L
Step 10/20 (50.00%) | loss 8.4396105774602503 6.9269 | lrm 1.0 | dt 4595.83ms 5930.69ms | tps 114,078 | mem 12.241 GB | shard 0 | time 00:01:17 | eta 00:00:59

MY-NANOCHAT FA3=True FP8=False
Step 10/20 (50.00%) | loss 8.4250608740851103 7.0665 | lrm 1.0 | dt 4572.83ms 5885.13ms | tps 114,652 | mem 12.241 GB | shard 0 | time 00:01:16 | eta 00:00:58

MY-NANOCHAT FA3=True FP8=True
Step 10/20 (50.00%) | loss nan nan | lrm 1.0 | dt 4097.53ms 5062.09ms | tps 127,952 | mem 16.377 GB | shard 0 | time 00:01:04 | eta 00:00:50



NANOCHAT FA3=False FP8=False
step 00010/00020 (50.00%) | loss: 7.538907 | lrm: 1.00 | dt: 4548.13ms | tok/sec: 115,275 | mfu: 72.44 | epoch: 1 | total time: 0.00m

NANOCHAT FA3=True FP8=False
step 00010/00020 (50.00%) | loss: 7.538950 | lrm: 1.00 | dt: 4512.59ms | tok/sec: 116,183 | mfu: 73.01 | epoch: 1 | total time: 0.00m

NANOCHAT FA3=True FP8=True
step 00010/00020 (50.00%) | loss: nan | lrm: 1.00 | dt: 4025.19ms | tok/sec: 130,251 | mfu: 81.85 | epoch: 1 | total time: 0.00m
```

