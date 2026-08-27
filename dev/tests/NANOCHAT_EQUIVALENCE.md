# Nanochat Bit Parity

For bit-equivalence testing, apply this patch on Nanochat side:

```
git clone git@github.com:karpathy/nanochat.git
cd nanochat
git checkout 92d63d4
git apply ../nanorepro/dev/tests/nanochat_equivalence.patch
```

Make sure both repos are setup (`uv sync`, data, tokenizers) run corresponding tests (edit CUDA_VISIBLE_DEVICES as needed):

```
# Nanochat-repro
uv run ./dev/tests/test_equivalence_d4.sh
uv run ./dev/tests/test_equivalence_sft_d4.sh

# Nanochat
uv run ./test_equivalence_d4.sh
uv run ./test_equivalence_sft_d4.sh
```

On success, these should match exactly:

- BPB on step 0 and last step, to full printed decimal places
- MD5 sum of saved `.pt` files, as printed to terminal

Last tested nanorepro `db84ab12` vs nanochat `92d63d4e`; both d4 and sft_d4 pass on 4x3090.