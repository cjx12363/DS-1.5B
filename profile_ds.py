import sys; sys.path.insert(0, r'E:\cjx12363\LLM4CP-DS')
import torch, numpy as np, time
DEVICE = 'cuda:0'
BATCH, PREV_LEN, PRED_LEN = 256, 16, 4
ENC_IN = 48 * 4 * 4 * 4

from models.LLM4CP import Model
print('Loading DS-1.5B...')
model = Model(llm_type='deepseek-1.5b', teacher_type=None, llm_layers=6,
    use_lora=True, lora_r=4, lora_alpha=8, lora_dropout=0.1, use_kd=False,
    d_ff=1536, d_model=1536, pred_len=PRED_LEN, prev_len=PREV_LEN,
    use_gpu=1, gpu_id=0, K=48, UQh=4, UQv=4, BQh=1, BQv=1,
    res_layers=4, res_dim=64, patch_size=4).to(DEVICE).eval()

tp = sum(p.nelement() for p in model.parameters())
tt = sum(p.nelement() for p in model.parameters() if p.requires_grad)
lora_params = sum(p.nelement() for n, p in model.named_parameters() if 'lora' in n.lower())
print(f'Total: {tp/1e6:.2f}M, Trainable: {tt/1e6:.2f}M, LoRA: {lora_params/1e6:.4f}M')

try:
    from fvcore.nn import FlopCountAnalysis
    inp = torch.randn(1, PREV_LEN, ENC_IN).to(DEVICE)
    flops = FlopCountAnalysis(model, inp).total()
    print(f'FLOPs: {flops/1e9:.2f} G')
except Exception as e:
    print(f'FLOPs: N/A ({e})')

print('Measuring inference time...')
inp = torch.randn(BATCH, PREV_LEN, ENC_IN).to(DEVICE)
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
with torch.no_grad():
    for _ in range(5):
        _ = model(inp, None, None, None)
    torch.cuda.synchronize()
    starter.record()
    for _ in range(20):
        _ = model(inp, None, None, None)
    ender.record()
    torch.cuda.synchronize()
inf_time = starter.elapsed_time(ender) / 20
print(f'Inference: {inf_time:.3f} ms (batch={BATCH})')
