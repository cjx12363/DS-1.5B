"""端到端 SE 评估: MIMO MRT+MRC (Nr=4, pol=2 → Nr_eff=8)"""
import argparse, os, numpy as np, torch
from eval_utils import *
from models.LLM4CP import Model

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='tdd', choices=['tdd', 'fdd'])
parser.add_argument('--scenario', type=str, default='UMa', choices=['UMa', 'UMi'])
parser.add_argument('--snr_dl', type=int, default=10)
parser.add_argument('--max_samples', type=int, default=100)
parser.add_argument('--bs', type=int, default=16)
args = parser.parse_args()

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
P, L, K, Nt = 16, 4, 48, 16
tag = 'D' if args.mode == 'fdd' else 'U'

his_raw = load_mat(f'./data/test/{args.scenario}_H_U_his_test.mat', 'H_U_his')
pre_raw = load_mat(f'./data/test/{args.scenario}_H_{tag}_pre_test.mat', f'H_{tag}_pre')

ckpt = f'./Weights/{"U2D_" if args.mode=="fdd" else ""}LLM4CP_DS.pth'
model = Model(llm_type='deepseek-1.5b', use_kd=False, use_lora=True, d_ff=1536, d_model=1536,
              pred_len=L, prev_len=P, K=K, UQh=4, UQv=4, BQh=1, BQv=1, use_gpu=1, gpu_id=0).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
print(f'{args.mode.upper()} {args.scenario} SNR={args.snr_dl}dB | Nt=16 Nr_eff=4×2=8 | MRT+MRC')

results = []
for sp in range(10):
    hs, ps = his_raw[[sp]], pre_raw[[sp]]
    n = min(args.max_samples, hs.shape[1])
    hs, ps = hs[:, :n], ps[:, :n]

    hc, pc = collapse_for_model(hs), collapse_for_model(ps)
    std = np.sqrt(np.std(np.abs(hc)**2) + 1e-12)

    # 模型预测
    inp = to_real(hc / std)
    preds = []
    with torch.no_grad():
        for s in range(0, n, args.bs):
            e = min(s + args.bs, n)
            preds.append(model(torch.tensor(inp[s:e]).to(device), None, None, None).cpu().numpy())
    pred = to_complex(np.concatenate(preds, axis=0) * std, n, L, K, Nt)

    # Perfect CSI (归一化→实数→复数→反归一化)
    perf = to_complex(to_real(pc / std), n, L, K, Nt) * std

    # 真 MIMO 信道 + NoPred
    Ht = full_mimo_channel(ps)                          # (n, L, K, Nt, 8)
    Hn = full_mimo_channel(hs)[:, -1:, :, :, :].repeat(L, axis=1)

    snr_lin = 10 ** (args.snr_dl / 10)
    se_p, se_d, se_n = [], [], []
    for t in range(L):
        for b in range(n):
            wp = mrt_precoder(perf[b, t])
            wd = mrt_precoder(pred[b, t])
            wn = mrt_precoder(Hn[b, t])
            nv = noise_calibration(Ht[b, t], wp, snr_lin)
            se_p.append(mrc_se(Ht[b, t], wp, nv))
            se_d.append(mrc_se(Ht[b, t], wd, nv))
            se_n.append(mrc_se(Ht[b, t], wn, nv))

    r = {'speed': (sp+1)*10, 'se_p': np.mean(se_p), 'se_d': np.mean(se_d), 'se_n': np.mean(se_n)}
    r['ds%'] = r['se_d']/r['se_p']*100; r['np%'] = r['se_n']/r['se_p']*100
    results.append(r)
    print(f"  {(sp+1)*10:>3d}km/h | Perf={r['se_p']:.3f} | DS={r['se_d']:.3f} ({r['ds%']:.1f}%) | NP={r['se_n']:.3f} ({r['np%']:.1f}%)")

avg = {k: np.mean([r[k] for r in results]) for k in ['se_p', 'se_d', 'se_n', 'ds%', 'np%']}
print(f"\n  Avg | Perf={avg['se_p']:.4f} | DS={avg['se_d']:.4f} ({avg['ds%']:.1f}%) | NP={avg['se_n']:.4f} ({avg['np%']:.1f}%)")

os.makedirs('result', exist_ok=True)
with open(f'result/e2e_{args.mode}_{args.scenario}_SNR{args.snr_dl}.csv', 'w') as f:
    f.write('Speed,Perfect_SE,DS_SE,DS_pct,NoPred_SE,NP_pct\n')
    for r in results:
        f.write(f"{r['speed']},{r['se_p']:.6f},{r['se_d']:.6f},{r['ds%']:.2f},{r['se_n']:.6f},{r['np%']:.2f}\n")
print('Done.')
