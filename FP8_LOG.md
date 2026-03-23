

## Baseline bf16

```
root@C.33395913:/workspace/my-nanochat$ uv run ./test_speed_d12_solo.sh 
Init: ddp=False ddp_rank=0, ddp_local_rank=0, ddp_world_size=1, ddp_master=True, device='cuda'
Init: FP8 Summary:
transformer.h.0.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.0.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.0.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.0.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.0.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.0.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.1.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.1.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.1.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.1.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.1.attn.ve_gate 32 6 torch.float32 cuda:0 False
transformer.h.1.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.1.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.2.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.2.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.2.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.2.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.2.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.2.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.3.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.3.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.3.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.3.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.3.attn.ve_gate 32 6 torch.float32 cuda:0 False
transformer.h.3.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.3.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.4.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.4.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.4.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.4.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.4.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.4.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.5.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.5.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.5.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.5.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.5.attn.ve_gate 32 6 torch.float32 cuda:0 False
transformer.h.5.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.5.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.6.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.6.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.6.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.6.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.6.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.6.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.7.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.7.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.7.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.7.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.7.attn.ve_gate 32 6 torch.float32 cuda:0 False
transformer.h.7.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.7.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.8.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.8.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.8.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.8.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.8.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.8.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.9.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.9.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.9.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.9.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.9.attn.ve_gate 32 6 torch.float32 cuda:0 False
transformer.h.9.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.9.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.10.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.10.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.10.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.10.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.10.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.10.mlp.c_proj 3072 768 torch.float32 cuda:0 True
transformer.h.11.attn.c_q 768 768 torch.float32 cuda:0 True
transformer.h.11.attn.c_k 768 768 torch.float32 cuda:0 True
transformer.h.11.attn.c_v 768 768 torch.float32 cuda:0 True
transformer.h.11.attn.c_proj 768 768 torch.float32 cuda:0 True
transformer.h.11.attn.ve_gate 32 6 torch.float32 cuda:0 False
transformer.h.11.mlp.c_fc 768 3072 torch.float32 cuda:0 True
transformer.h.11.mlp.c_proj 3072 768 torch.float32 cuda:0 True
lm_head 768 65536 torch.float32 cuda:0 True
  Eligible for FP8: 73/79 linear layers
Init: Model info:
  block_size=2048
  vocab_size=65536
  depth=12
  model_dim=768
  num_heads=6
Init: Scaling info:
  Scaling params (matrices + lm_head): 135,267,456
  Target tokens (scaling_params * target_param_data_ratio): 1,420,308,288
Init: Calculated total_batch_size=524288 based on Power Lines scaling with target_token_ratio=1.00. Proposed batch size before rounding: 524288.00
Init: Scaled weight decay: 0.2 -> 0.2
Init: Training hyperparameters:
  micro_batch=16
  total_batch_size=524288
  grad_accum=16
Init: Using user-provided num_iterations=20 without scaling.
Init: Train dataloader initialised with shards 0 - 9
Init: Eval BPB every 10 steps, eval_steps=16
Init: Eval dataloader initialised with shards 1822 - 1822
BPB Eval 0 | BPB 3.25722746753737 | nats 5810483.0 | bytes 2573586.0
Step 0/20 (0.00%) | loss 11.0907230377197248 11.0906 | lrm 1.0 | dt 22500.33ms 22500.33ms | tps 23,301 | mem 17.761 GB | time 00:00:22 | eta 00:07:30
Step 1/20 (5.00%) | loss 11.3602575000963704 11.6089 | lrm 1.0 | dt 4799.03ms 13183.85ms | tps 109,248 | mem 19.763 GB | time 00:00:27 | eta 00:04:10
Step 2/20 (10.00%) | loss 11.2038558382829621 10.9400 | lrm 1.0 | dt 4748.54ms 10071.19ms | tps 110,410 | mem 19.763 GB | time 00:00:32 | eta 00:03:01
Step 3/20 (15.00%) | loss 11.0124004255307533 10.5188 | lrm 1.0 | dt 4757.86ms 8526.17ms | tps 110,194 | mem 19.763 GB | time 00:00:36 | eta 00:02:24
Step 4/20 (20.00%) | loss 10.5560913826967422 9.1360 | lrm 1.0 | dt 4794.50ms 7614.92ms | tps 109,352 | mem 19.763 GB | time 00:00:41 | eta 00:02:01
Step 5/20 (25.00%) | loss 10.0785717610075025 8.3707 | lrm 1.0 | dt 4791.72ms 7012.39ms | tps 109,415 | mem 19.763 GB | time 00:00:46 | eta 00:01:45
Step 6/20 (30.00%) | loss 9.6486493672257154 7.9301 | lrm 1.0 | dt 4818.79ms 6591.92ms | tps 108,800 | mem 19.763 GB | time 00:00:51 | eta 00:01:32
Step 7/20 (35.00%) | loss 9.6702260891186498 9.7540 | lrm 1.0 | dt 4822.32ms 6281.21ms | tps 108,721 | mem 19.763 GB | time 00:00:56 | eta 00:01:21
Step 8/20 (40.00%) | loss 9.4278780700802134 8.0730 | lrm 1.0 | dt 4835.85ms 6045.26ms | tps 108,416 | mem 19.763 GB | time 00:01:00 | eta 00:01:12
Step 9/20 (45.00%) | loss 9.1822623982886800 7.7649 | lrm 1.0 | dt 4836.87ms 5859.74ms | tps 108,393 | mem 19.763 GB | time 00:01:05 | eta 00:01:04
BPB Eval 10 | BPB 2.18135432330788 | nats 3891261.0 | bytes 2573586.0
Step 10/20 (50.00%) | loss 8.9277116496838964 7.4419 | lrm 1.0 | dt 4850.41ms 5712.64ms | tps 108,091 | mem 19.763 GB | time 00:01:10 | eta 00:00:57
Step 11/20 (55.00%) | loss 8.7191523045473911 7.2868 | lrm 0.9 | dt 4877.37ms 5596.24ms | tps 107,494 | mem 19.763 GB | time 00:01:15 | eta 00:00:50
Step 12/20 (60.00%) | loss 8.5392656414967494 7.1801 | lrm 0.8 | dt 4846.29ms 5495.69ms | tps 108,183 | mem 19.763 GB | time 00:01:20 | eta 00:00:43
Step 13/20 (65.00%) | loss 8.3413915197007764 7.0497 | lrm 0.7 | dt 4940.62ms 5423.72ms | tps 106,117 | mem 19.763 GB | time 00:01:25 | eta 00:00:37
Step 14/20 (70.00%) | loss 8.1672404822793769 6.9057 | lrm 0.6 | dt 5268.96ms 5404.23ms | tps 99,505 | mem 19.763 GB | time 00:01:30 | eta 00:00:32
Step 15/20 (75.00%) | loss 7.9992455584644446 6.7616 | lrm 0.5 | dt 4978.13ms 5351.93ms | tps 105,318 | mem 19.763 GB | time 00:01:35 | eta 00:00:26
Step 16/20 (80.00%) | loss 7.8409244480159463 6.7209 | lrm 0.4 | dt 5232.49ms 5337.59ms | tps 100,198 | mem 19.763 GB | time 00:01:40 | eta 00:00:21
Step 17/20 (85.00%) | loss 7.7158272483142154 6.6897 | lrm 0.3 | dt 5172.95ms 5318.22ms | tps 101,351 | mem 19.763 GB | time 00:01:45 | eta 00:00:15
Step 18/20 (90.00%) | loss 7.5904656313279979 6.6295 | lrm 0.2 | dt 5815.28ms 5375.69ms | tps 90,156 | mem 19.763 GB | time 00:01:51 | eta 00:00:10
Step 19/20 (95.00%) | loss 7.4688123548524459 6.6069 | lrm 0.1 | dt 5478.98ms 5387.45ms | tps 95,690 | mem 19.763 GB | time 00:01:57 | eta 00:00:05
BPB Eval 20 | BPB 1.93235325601119 | nats 3447074.5 | bytes 2573586.0
Task               hellaswag_zeroshot (multiple_choice, 0-shot) | dt 15.0s | acc 0.2520 | centered_acc 0.0027
Task                         jeopardy (language_modeling, 10-shot) | dt 12.0s | acc 0.0000 | centered_acc 0.0000
Task             bigbench_qa_wikidata (language_modeling, 10-shot) | dt 12.3s | acc 0.0000 | centered_acc 0.0000
Task                         arc_easy (multiple_choice, 10-shot) | dt 15.5s | acc 0.2720 | centered_acc 0.0293
Task                    arc_challenge (multiple_choice, 10-shot) | dt 16.9s | acc 0.2300 | centered_acc -0.0267
Task                             copa (multiple_choice, 0-shot) | dt 2.2s | acc 0.4900 | centered_acc -0.0200
Task                   commonsense_qa (multiple_choice, 10-shot) | dt 16.2s | acc 0.1700 | centered_acc -0.0375
Task                             piqa (multiple_choice, 10-shot) | dt 13.3s | acc 0.5160 | centered_acc 0.0320
Task                      openbook_qa (multiple_choice, 0-shot) | dt 11.2s | acc 0.2020 | centered_acc -0.0640
Task                   lambada_openai (language_modeling, 0-shot) | dt 13.1s | acc 0.0000 | centered_acc 0.0000
Task                        hellaswag (multiple_choice, 10-shot) | dt 22.6s | acc 0.2520 | centered_acc 0.0027
Task                         winograd (schema, 0-shot) | dt 5.9s | acc 0.5201 | centered_acc 0.0403
Task                       winogrande (schema, 0-shot) | dt 11.0s | acc 0.4760 | centered_acc -0.0480
Task          bigbench_dyck_languages (language_modeling, 10-shot) | dt 12.2s | acc 0.0000 | centered_acc 0.0000
Task                 agi_eval_lsat_ar (multiple_choice, 3-shot) | dt 8.9s | acc 0.2261 | centered_acc 0.0326
Task           bigbench_cs_algorithms (language_modeling, 10-shot) | dt 12.0s | acc 0.0000 | centered_acc 0.0000
Task               bigbench_operators (language_modeling, 10-shot) | dt 5.2s | acc 0.0000 | centered_acc 0.0000
Task       bigbench_repeat_copy_logic (language_modeling, 10-shot) | dt 0.8s | acc 0.0000 | centered_acc 0.0000
Task                            squad (language_modeling, 10-shot) | dt 21.9s | acc 0.0000 | centered_acc 0.0000
Task                             coqa (language_modeling, 0-shot) | dt 14.4s | acc 0.0000 | centered_acc 0.0000
Task                            boolq (multiple_choice, 10-shot) | dt 20.5s | acc 0.3720 | centered_acc -0.6526
Task bigbench_language_identification (multiple_choice, 10-shot) | dt 24.8s | acc 0.2380 | centered_acc 0.1617
CORE 20 | core metric -0.02488697755515 | dt 287.87s
<|bos|>The capital of France is a new study, and the same, and the same, and the same,
<|bos|>The chemical symbol of gold is a new study, and the same, and the same, and the same,
<|bos|>If yesterday was Friday, then tomorrow will be used to the same, and the same, and the same, and the same
<|bos|>The opposite of hot is a new study, and the first, and the first, and the same,
<|bos|>The planets of the solar system are: The first, and the same, and the first, and the same, and
<|bos|>My favorite color is a new study, and the same, and the first, and the same,
<|bos|>If 5*x + 3 = 13, then x is a new study, and the same, and the same, and the same,
Mem rank 0: 4452.9MiB, Res: 21982.0MiB, Max: 20237.3MiB
```