import torch
from transformer import Transformer

# Tiny vocab, tiny model
model = Transformer(
    encoder_num=2, decoder_num=2,
    num_heads=2, d_model=32, d_ff=64,
    vocab_size=100, max_len=20
)

B, S, T = 2, 8, 6
src = torch.randint(0, 100, (B, S))
tgt = torch.randint(0, 100, (B, T))

# Causal mask
def causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)

tgt_mask = causal_mask(T)

out = model(src, tgt, tgt_mask=tgt_mask)

# 1. Shape check
assert out.shape == (B, T, 100), f"Wrong shape: {out.shape}"

# 2. Attention weights sum to 1 — you need to return them from MHA to check this
# Skip for now unless you add a return

# 3. Gradients flow
loss = out.sum()
loss.backward()
for name, p in model.named_parameters():
    assert p.grad is not None, f"No grad: {name}"
    assert not torch.isnan(p.grad).any(), f"NaN grad: {name}"

print("All checks passed")

# 4. Overfit single batch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

src_small  = torch.randint(0, 100, (2, 8))
tgt_small  = torch.randint(0, 100, (2, 6))
label      = torch.randint(0, 100, (2, 6))  # what decoder should predict

for step in range(200):
    optimizer.zero_grad()
    out = model(src_small, tgt_small, tgt_mask=causal_mask(6))
    loss = criterion(out.view(-1, 100), label.view(-1))
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == 199:
        print(f"step {step} loss {loss.item():.4f}")

# Should reach < 0.1 by step 200