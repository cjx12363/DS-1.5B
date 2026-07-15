import sys; sys.path.insert(0, r'E:\cjx12363\LLM4CP-DS')
import torch, time
DEVICE = 'cuda:0'
BATCH, PREV_LEN = 256, 16
ENC_IN = 48 * 4 * 4 * 4

from models.LLM4CP import Model
print('Loading...')
model = Model(llm_type='deepseek-1.5b', teacher_type=None, llm_layers=6,
    use_lora=True, lora_r=4, lora_alpha=8, lora_dropout=0.1, use_kd=False,
    d_ff=1536, d_model=1536, pred_len=4, prev_len=16,
    use_gpu=1, gpu_id=0, K=48, UQh=4, UQv=4, BQh=1, BQv=1,
    res_layers=4, res_dim=64, patch_size=4).to(DEVICE).eval()

inp = torch.randn(BATCH, PREV_LEN, ENC_IN).to(DEVICE)
print('Warmup...')
with torch.no_grad():
    for _ in range(3):
        _ = model(inp, None, None, None)
torch.cuda.synchronize()
print('Benchmarking...')
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(inp, None, None, None)
torch.cuda.synchronize()
t = (time.time() - t0) / 10 * 1000
print(f'Inference: {t:.1f} ms (batch={BATCH})')
