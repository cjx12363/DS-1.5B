"""可达速率评估: MIMO MRT+MRC, 每子载波速率曲线"""
import argparse, os, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eval_utils import *
from models.LLM4CP import Model

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='tdd', choices=['tdd', 'fdd'])
parser.add_argument('--scenario', type=str, default='UMa', choices=['UMa', 'UMi'])
parser.add_argument('--snr', type=float, default=10.0)
parser.add_argument('--bs', type=int, default=64)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
P, L, K, Nt = 16, 4, 48, 16
tag = 'D' if args.mode == 'fdd' else 'U'

his_raw = load_mat(f'./data/test/{args.scenario}_H_U_his_test.mat', 'H_U_his')
pre_raw = load_mat(f'./data/test/{args.scenario}_H_{tag}_pre_test.mat', f'H_{tag}_pre')

ckpt = f'./Weights/{"U2D_" if args.mode=="fdd" else ""}LLM4CP_DS.pth'
model = Model(llm_type='deepseek-1.5b', use_kd=False, use_lora=True, d_ff=1536, d_model=1536,
              pred_len=L, prev_len=P, K=K, UQh=4, UQv=4, BQh=1, BQv=1, use_gpu=1, gpu_id=0).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
print(f'{args.mode.upper()} {args.scenario} SNR={args.snr}dB | Nt=16 Nr_eff=8 | MRT+MRC')

rate_p, rate_d, rate_n = [], [], []
for sp in range(10):
    hs, ps = his_raw[[sp]], pre_raw[[sp]]
    n = hs.shape[1]
    hc, pc = collapse_for_model(hs), collapse_for_model(ps)
    std = np.sqrt(np.std(np.abs(hc)**2) + 1e-12)

    inp = to_real(hc / std)
    preds = []
    with torch.no_grad():
        for s in range(0, n, args.bs):
            e = min(s + args.bs, n)
            preds.append(model(torch.tensor(inp[s:e]).to(device), None, None, None).cpu().numpy())
    pred = to_complex(np.concatenate(preds, axis=0) * std, n, L, K, Nt)
    perf = to_complex(to_real(pc / std), n, L, K, Nt) * std
    Ht = full_mimo_channel(ps)
    Hn = full_mimo_channel(hs)[:, -1:, :, :, :].repeat(L, axis=1)

    snr_lin = 10**(args.snr/10)
    for t in range(L):
        H = Ht[:, t]  # (n, K, Nt, 8)
        wp = mrt_precoder(perf[:, t])  # (n, K, Nt)
        wd = mrt_precoder(pred[:, t])
        wn = mrt_precoder(Hn[:, t])

        # 全局噪声校准
        gain_p = np.mean([noise_calibration(H[b], wp[b], snr_lin)*snr_lin for b in range(n)])
        nv = gain_p / snr_lin

        for b in range(n):
            rate_p.append(np.log2(1+mrc_sinr(H[b], wp[b], nv)))
            rate_d.append(np.log2(1+mrc_sinr(H[b], wd[b], nv)))
            rate_n.append(np.log2(1+mrc_sinr(H[b], wn[b], nv)))
    print(f'  speed {sp+1}/10 done')

rp = np.mean(np.concatenate(rate_p), axis=0)
rd = np.mean(np.concatenate(rate_d), axis=0)
rn = np.mean(np.concatenate(rate_n), axis=0)
print(f'\n=== {args.mode.upper()} / {args.scenario} ===')
print(f'  Perfect={np.mean(rp):.4f}  Pred={np.mean(rd):.4f}  NoPred={np.mean(rn):.4f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(K), rp, 'g-', lw=2, label=f'Perfect ({np.mean(rp):.2f})')
ax1.plot(range(K), rd, 'b--', lw=2, label=f'Predicted ({np.mean(rd):.2f})')
ax1.plot(range(K), rn, 'r:', lw=2, label=f'No-Pred ({np.mean(rn):.2f})')
ax1.set_xlabel('Subcarrier'); ax1.set_ylabel('Rate (bits/s/Hz)')
ax1.set_title(f'Per-Subcarrier Rate ({args.mode.upper()}/{args.scenario})')
ax1.legend(); ax1.grid(alpha=0.3)
for bar, val, c in zip(['Perfect', 'Predicted', 'No-Pred'], [np.mean(rp), np.mean(rd), np.mean(rn)], ['green', 'steelblue', 'tomato']):
    b = ax2.bar(bar, val, color=c, edgecolor='black', lw=0.8)
    ax2.text(b[0].get_x()+b[0].get_width()/2, val+0.02, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean rate (bits/s/Hz)'); ax2.set_title(f'Average Rate'); ax2.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'rate_{args.mode}_{args.scenario}.png', dpi=150); plt.close()
np.savetxt(f'rate_{args.mode}_{args.scenario}.csv', np.stack([rp, rd, rn], axis=0).T, delimiter=',', header='perfect,predicted,nopred', comments='')
print('Done.')
