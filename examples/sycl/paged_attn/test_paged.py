import torch
import paged_attention

import math

import torch.nn.functional as F
from einops import rearrange, repeat
from time import perf_counter_ns

torch.set_printoptions(threshold=float('inf'))

def assert_close_verbose(a, b, rtol=1e-2, atol=1e-1):
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Compute absolute and relative differences
    diff = torch.abs(a - b)
    tol = atol + rtol * torch.abs(b)

    # Find mismatches
    mismatch_mask = diff > tol
    if mismatch_mask.any():
        idx = mismatch_mask.nonzero(as_tuple=False)
        for i in idx:
            coord = tuple(i.tolist())
            val_a = a[coord].item()
            val_b = b[coord].item()
            d = diff[coord].item()
            t = tol[coord].item()
            print(f"Mismatch at {coord}: a={val_a}, b={val_b}, diff={d}, tol={t}")
        # raise AssertionError(f"Tensors are not close. {mismatch_mask.sum().item()} elements differ.")
    else:
        print("Tensors are close within tolerance.")

def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


device = "xpu"
dtype = torch.bfloat16

num_heads_q = 16
num_heads_kv = 2
seqlen_q = 1
num_blocks = 160
block_size = 64
head_size = 128
seqlen_kv = num_blocks * block_size
max_blocks_per_seq = num_blocks
group_heads = num_heads_q // num_heads_kv
num_partitions = seqlen_kv // 512

query = torch.rand((seqlen_q, num_heads_q, head_size), dtype=dtype, device=device)
key_cache = torch.rand((num_blocks, block_size, num_heads_kv, head_size), dtype=dtype, device=device)
value_cache = torch.rand((num_blocks, block_size, num_heads_kv, head_size), dtype=dtype, device=device)
out = torch.zeros_like(query).float()

block_tables = torch.randint(0, max_blocks_per_seq, (seqlen_q, max_blocks_per_seq), dtype=torch.int, device=device)

max_logits = torch.zeros((seqlen_q, num_partitions, num_heads_q), dtype=torch.float32, device=device)
exp_sums = torch.zeros_like(max_logits)
temp_out = torch.zeros((seqlen_q, num_heads_q, num_partitions, head_size), dtype=torch.float, device=device)
scores = torch.zeros((seqlen_q, num_heads_kv, group_heads, seqlen_kv), dtype=torch.float, device=device)

query_ref = query.view(1, seqlen_q, num_heads_q, head_size).contiguous()
k_cache_ref = rearrange(
    key_cache[block_tables.to(dtype=torch.long).flatten()],
    "(b nblocks) block_size ... -> b (nblocks block_size) ...",
    b=seqlen_q,
)[:, :seqlen_kv]
v_cache_ref = rearrange(
    value_cache[block_tables.to(dtype=torch.long).flatten()],
    "(b nblocks) block_size ... -> b (nblocks block_size) ...",
    b=seqlen_q,
)[:, :seqlen_kv]

sm_scale = 1. / math.sqrt(head_size)

# warm up
for i in range(10):
    paged_attention.run(
        max_logits,
        exp_sums,
        temp_out,
        out,
        scores,
        query,
        key_cache,
        value_cache,
        block_tables,
        sm_scale,
        block_size,
        seqlen_kv
    )

start_counter = perf_counter_ns()
for i in range(1000):
    paged_attention.run(
        max_logits,
        exp_sums,
        temp_out,
        out,
        scores,
        query,
        key_cache,
        value_cache,
        block_tables,
        sm_scale,
        block_size,
        seqlen_kv
    )
end_counter = perf_counter_ns()

exec_time = (end_counter - start_counter) / 1000 * 1e-3;
total_kv_size = key_cache.numel() * key_cache.element_size() * 2
print(
    f"Time: {exec_time:.2f} us, "
    f"Total KV size: {total_kv_size / (1024 * 1024):.2f} MB, "
    f"Bandwidth: {total_kv_size / exec_time / (1000):.2f} GB/s"
)


out_ref, scores_ref = attention_ref(query_ref, k_cache_ref, v_cache_ref)

out_ref_1, scores_ref_1 = attention_ref(query_ref, k_cache_ref[:, :512, :, :], v_cache_ref[:, :512, :, :])
out_ref_2, scores_ref_2 = attention_ref(query_ref, k_cache_ref[:, 512:, :, :], v_cache_ref[:, 512:, :, :])

scores_ref = scores_ref.view(seqlen_q, num_heads_kv, group_heads, seqlen_kv)
out_ref = out_ref.view(seqlen_q, num_heads_q, head_size)
out_ref_1 = out_ref_1.view(seqlen_q, num_heads_q, head_size)
out_ref_2 = out_ref_2.view(seqlen_q, num_heads_q, head_size)

# assert_close_verbose(scores.max(dim=-1)[0].flatten().cpu().float(), max_logits.flatten().cpu().float())
# assert_close_verbose(scores.sum(dim=-1)[0].flatten().cpu().float(), exp_sums.flatten().cpu().float())
# assert_close_verbose(scores_ref.cpu().float(), scores.cpu().float())
# assert_close_verbose(out_ref_1.cpu().float(), temp_out[:, :, 0, :].cpu().float())
# assert_close_verbose(out_ref_2.cpu().float(), temp_out[:, :, 1, :].cpu().float())
