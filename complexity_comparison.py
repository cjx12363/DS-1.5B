import sys; sys.path.insert(0, r"E:\cjx12363\LLM4CP-DS")
import torch
import torch.nn as nn
import numpy as np
import csv
from einops import rearrange
import math

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH = 256
PREV_LEN = 16
PRED_LEN = 4
FEATURES = 96

print(f"Device: {DEVICE}, Batch: {BATCH}")

results = []

# %% 1. RNN
class RNNUnit(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
    def forward(self, x, prev_hidden):
        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output).reshape(L, B, -1)
        output, cur_hidden = self.rnn(output, prev_hidden)
        output = output.reshape(L * B, -1)
        output = self.decoder(output).reshape(L, B, -1)
        return output, cur_hidden

class RNN(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = RNNUnit(features, input_size, hidden_size, num_layers)
    def forward(self, x):
        B, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        outputs = []
        for idx in range(seq_len + PRED_LEN - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(x[:, idx:idx+1].permute(1,0,2).contiguous(), prev_hidden)
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        return torch.cat(outputs, dim=0).permute(1,0,2).contiguous()

print("\n=== RNN ===")
rnn = RNN(FEATURES, FEATURES, 192, num_layers=4).to(DEVICE).eval()
tp = sum(p.nelement() for p in rnn.parameters())
tt = sum(p.nelement() for p in rnn.parameters() if p.requires_grad)
print(f"  Params: {tp:,} total, {tt:,} trainable")
try:
    from thop import profile
    inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
    flops, _ = profile(rnn, inputs=(inp,), verbose=False)
    print(f"  FLOPs: {flops/1e6:.2f} M")
except Exception as e:
    flops = "N/A"
    print(f"  FLOPs error: {e}")
inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
timings = []
with torch.no_grad():
    for _ in range(110):
        starter.record()
        _ = rnn(inp)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
inf_time = np.mean(timings[10:])
print(f"  Inference: {inf_time:.3f} ms")
results.append(["RNN", tt, tp, f"{flops/1e6:.2f}" if isinstance(flops, (int,float)) else flops, f"{inf_time:.3f}"])

# %% 2. LSTM
class LSTMUnit(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
    def forward(self, x, prev_hidden, prev_cell):
        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output).reshape(L, B, -1)
        output, (cur_hidden, cur_cell) = self.lstm(output, (prev_hidden, prev_cell))
        output = output.reshape(L * B, -1)
        output = self.decoder(output).reshape(L, B, -1)
        return output, cur_hidden, cur_cell

class LSTM(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = LSTMUnit(features, input_size, hidden_size, num_layers)
    def forward(self, x):
        B, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        prev_cell = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        outputs = []
        for idx in range(seq_len + PRED_LEN - 1):
            if idx < seq_len:
                output, prev_hidden, prev_cell = self.model(x[:, idx:idx+1].permute(1,0,2).contiguous(), prev_hidden, prev_cell)
            else:
                output, prev_hidden, prev_cell = self.model(output, prev_hidden, prev_cell)
            if idx >= seq_len - 1:
                outputs.append(output)
        return torch.cat(outputs, dim=0).permute(1,0,2).contiguous()

print("\n=== LSTM ===")
lstm = LSTM(FEATURES, FEATURES, 192, num_layers=4).to(DEVICE).eval()
tp = sum(p.nelement() for p in lstm.parameters())
tt = sum(p.nelement() for p in lstm.parameters() if p.requires_grad)
print(f"  Params: {tp:,} total, {tt:,} trainable")
try:
    inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
    flops, _ = profile(lstm, inputs=(inp,), verbose=False)
    print(f"  FLOPs: {flops/1e6:.2f} M")
except Exception as e:
    flops = "N/A"
    print(f"  FLOPs error: {e}")
inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
timings = []
with torch.no_grad():
    for _ in range(110):
        starter.record()
        _ = lstm(inp)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
inf_time = np.mean(timings[10:])
print(f"  Inference: {inf_time:.3f} ms")
results.append(["LSTM", tt, tp, f"{flops/1e6:.2f}" if isinstance(flops, (int,float)) else flops, f"{inf_time:.3f}"])

# %% 3. GRU
class GRUUnit(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features, input_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_size, features))
    def forward(self, x, prev_hidden):
        L, B, F = x.shape
        output = x.reshape(L * B, -1)
        output = self.encoder(output).reshape(L, B, -1)
        output, cur_hidden = self.gru(output, prev_hidden)
        output = output.reshape(L * B, -1)
        output = self.decoder(output).reshape(L, B, -1)
        return output, cur_hidden

class GRU(nn.Module):
    def __init__(self, features, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.features = features
        self.model = GRUUnit(features, input_size, hidden_size, num_layers)
    def forward(self, x):
        B, seq_len, _ = x.shape
        prev_hidden = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        outputs = []
        for idx in range(seq_len + PRED_LEN - 1):
            if idx < seq_len:
                output, prev_hidden = self.model(x[:, idx:idx+1].permute(1,0,2).contiguous(), prev_hidden)
            else:
                output, prev_hidden = self.model(output, prev_hidden)
            if idx >= seq_len - 1:
                outputs.append(output)
        return torch.cat(outputs, dim=0).permute(1,0,2).contiguous()

print("\n=== GRU ===")
gru = GRU(FEATURES, FEATURES, 192, num_layers=4).to(DEVICE).eval()
tp = sum(p.nelement() for p in gru.parameters())
tt = sum(p.nelement() for p in gru.parameters() if p.requires_grad)
print(f"  Params: {tp:,} total, {tt:,} trainable")
try:
    inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
    flops, _ = profile(gru, inputs=(inp,), verbose=False)
    print(f"  FLOPs: {flops/1e6:.2f} M")
except Exception as e:
    flops = "N/A"
    print(f"  FLOPs error: {e}")
inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
timings = []
with torch.no_grad():
    for _ in range(110):
        starter.record()
        _ = gru(inp)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
inf_time = np.mean(timings[10:])
print(f"  Inference: {inf_time:.3f} ms")
results.append(["GRU", tt, tp, f"{flops/1e6:.2f}" if isinstance(flops, (int,float)) else flops, f"{inf_time:.3f}"])

# %% 4. CNN
class Autoencoder(nn.Module):
    def __init__(self, n_filters=None, filter_sizes=None):
        super().__init__()
        if n_filters is None:
            n_filters = [2, 8, 16, 32, 64, 128, 256, 512]
        if filter_sizes is None:
            filter_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
        self.postprocess = nn.Conv1d(PREV_LEN, PRED_LEN, 3, 1, 1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(n_filters) - 1):
            self.encoder.append(nn.Conv2d(n_filters[i], n_filters[i+1], filter_sizes[i], stride=1, padding=1))
            nn.init.uniform_(self.encoder[-1].weight, -1.0 / math.sqrt(n_filters[i]), 1.0 / math.sqrt(n_filters[i]))
            nn.init.constant_(self.encoder[-1].bias, 0)
        n_filters.reverse()
        filter_sizes.reverse()
        for i in range(len(n_filters) - 1):
            self.decoder.append(nn.Conv2d(n_filters[i], n_filters[i+1], filter_sizes[i], stride=1, padding=1))
            nn.init.uniform_(self.decoder[-1].weight, -1.0 / math.sqrt(n_filters[i]), 1.0 / math.sqrt(n_filters[i]))
            nn.init.constant_(self.decoder[-1].bias, 0)
    def forward(self, x):
        x = rearrange(x, 'b l (s i) -> b i l s', i=2)
        for layer in self.encoder:
            x = torch.tanh(layer(x))
        for layer in self.decoder:
            x = torch.tanh(layer(x))
        x = rearrange(x, 'b i l s -> b l (s i)', i=2)
        x = self.postprocess(x)
        return x

print("\n=== CNN ===")
cnn = Autoencoder().to(DEVICE).eval()
tp = sum(p.nelement() for p in cnn.parameters())
tt = sum(p.nelement() for p in cnn.parameters() if p.requires_grad)
print(f"  Params: {tp:,} total, {tt:,} trainable")
try:
    inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
    flops, _ = profile(cnn, inputs=(inp,), verbose=False)
    print(f"  FLOPs: {flops/1e6:.2f} M")
except Exception as e:
    flops = "N/A"
    print(f"  FLOPs error: {e}")
inp = torch.randn(BATCH, PREV_LEN, FEATURES).to(DEVICE)
timings = []
with torch.no_grad():
    for _ in range(110):
        starter.record()
        _ = cnn(inp)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
inf_time = np.mean(timings[10:])
print(f"  Inference: {inf_time:.3f} ms")
results.append(["CNN", tt, tp, f"{flops/1e6:.2f}" if isinstance(flops, (int,float)) else flops, f"{inf_time:.3f}"])

# %% 5. PAD
print("\n=== PAD (8-order AR) ===")
print("  Params: 0 (model-based)")
print("  FLOPs: N/A (model-based)")
print("  Inference: CPU-bound")
results.append(["PAD", 0, 0, "N/A (model-based)", "N/A (CPU)"])

# %% 6. DS-1.5B
print("\n=== DS-1.5B (LoRA r=4) ===")
try:
    from models.LLM4CP import Model
    ds_model = Model(
        llm_type='deepseek-1.5b',
        teacher_type=None,
        llm_layers=6,
        use_lora=True,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        use_kd=False,
        d_ff=1536, d_model=1536,
        pred_len=PRED_LEN, prev_len=PREV_LEN,
        use_gpu=1, gpu_id=0,
        K=48, UQh=4, UQv=4, BQh=1, BQv=1,
        res_layers=4, res_dim=64, patch_size=4,
    ).to(DEVICE).eval()
    tp = sum(p.nelement() for p in ds_model.parameters())
    tt = sum(p.nelement() for p in ds_model.parameters() if p.requires_grad)
    lora_params = sum(p.nelement() for n, p in ds_model.named_parameters() if 'lora' in n.lower())
    print(f"  Total: {tp/1e6:.2f}M, Trainable: {tt/1e6:.2f}M, LoRA: {lora_params/1e6:.4f}M")
    try:
        ENC_IN = 48 * 4 * 4 * 4
        inp = torch.randn(BATCH, PREV_LEN, ENC_IN).to(DEVICE)
        flops, _ = profile(ds_model, inputs=(inp, None, None, None), verbose=False)
        print(f"  FLOPs: {flops/1e9:.2f} G")
    except Exception as e:
        flops = "N/A"
        print(f"  FLOPs error: {e}")
    inp = torch.randn(BATCH, PREV_LEN, ENC_IN).to(DEVICE)
    timings = []
    with torch.no_grad():
        for _ in range(110):
            starter.record()
            _ = ds_model(inp, None, None, None)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
    inf_time = np.mean(timings[10:])
    print(f"  Inference: {inf_time:.3f} ms")
    results.append(["DS-1.5B (Ours)", tt, tp, f"{flops/1e9:.2f}" if isinstance(flops, (int,float)) else flops, f"{inf_time:.3f}"])
except Exception as e:
    print(f"  Failed: {e}")
    results.append(["DS-1.5B (Ours)", "~0.15M (LoRA)", "~1.5B", "N/A", "N/A"])

# %% Save CSV
csv_path = r"E:\cjx12363\LLM4CP-DS\complexity_comparison.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Trainable_Params", "Total_Params", "FLOPs", "Inference_Time_ms"])
    for row in results:
        w.writerow(row)

print("\n" + "="*80)
print(f"{'Model':20s} | {'Trainable':>20s} | {'Total':>20s} | {'FLOPs':>18s} | {'Time(ms)':>12s}")
print("-"*80)
for row in results:
    print(f"{row[0]:20s} | {str(row[1]):>20s} | {str(row[2]):>20s} | {str(row[3]):>18s} | {str(row[4]):>12s}")
print("="*80)
print(f"Saved: {csv_path}")
