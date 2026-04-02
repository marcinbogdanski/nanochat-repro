import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


input_moe = torch.zeros(16, 64, dtype=torch.bfloat16, device="cuda")
input_moe[:,0] = 1
weights_moe = torch.zeros(4, 64, 128, dtype=torch.bfloat16, device="cuda")  # 4 experts
weights_moe[0,0,0] = 0
weights_moe[1,0,0] = 1
weights_moe[2,0,0] = 2
weights_moe[3,0,0] = 3
expert_offsets = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device="cuda")

outputs = torch._grouped_mm(
    input=input_moe,
    mat2=weights_moe,
    offs=expert_offsets,
)

# tensor([0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.])
print(outputs[:,0])
print("Bye")
