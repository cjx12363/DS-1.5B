"""BER 仿真: MIMO MRT+MRC, QAM 调制"""
import argparse, os, numpy as np, torch
from eval_utils import *
from models.LLM4CP import Model

def qam_mod(bits, mo):
    M, nm = 2**mo, bits.shape[0]//mo
    bits = bits[:nm*mo].reshape(nm, mo)
    sm = int(np.sqrt(M))
    ri = np.sum([bits[:, i]<<i for i in range(mo//2)], axis=0)
    ii = np.sum([bits[:, i+mo//2]<<i for i in range(mo//2)], axis=0)
    return ((2*ri-sm+1) + 1j*(2*ii-sm+1)) / np.sqrt((M-1)*2/3)

def qam_demod(syms, mo):
    M, sm = 2**mo, int(np.sqrt(2**mo))
    syms *= np.sqrt((M-1)*2/3)
    rv = np.clip(np.round((syms.real+sm-1)/2), 0, sm-1).astype(int)
    iv = np.clip(np.round((syms.imag+sm-1)/2), 0, sm-1).astype(int)
    bits = np.zeros(len(syms)*mo, dtype=int)
    for i in range(mo//2): bits[i::mo] = (rv>>i)&1
    for i in range(mo//2): bits[mo//2+i::mo] = (iv>>i)&1
    return bits

def ber(tx_bits, rx_sym, mo, nv):
    n = np.sqrt(nv/2)*(np.random.randn(*rx_sym.shape)+1j*np.random.randn(*rx_sym.shape))
    return np.mean(tx_bits != qam_demod(rx_sym+n, mo))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='tdd', choices=['tdd', 'fdd'])
parser.add_argument('--scenario', type=str, default='UMa', choices=['UMa', 'UMi'])
parser.add_argument('--modulation', type=str, default='QPSK', choices=['QPSK', '16QAM', '64QAM'])
parser.add_argument('--snr_dl', type=int, default=10)
parser.add_argument('--max_samples', type=int, default=200)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--n_bits', type=int, default=100000)
args = parser.parse_args()

mo = {'QPSK': 2, '16QAM': 4, '64QAM': 6}[args.modulation]
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
print(f'{args.mode.upper()} {args.scenario} {args.modulation} SNR={args.snr_dl}dB | MRT+MRC')

results = []
for sp in range(10):
    hs, ps = his_raw[[sp]], pre_raw[[sp]]
    n = min(args.max_samples, hs.shape[1])
    hs, ps = hs[:, :n], ps[:, :n]

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

    snr_lin = 10**(args.snr_dl/10)
    bp, bd, bn, sp_s, sd_s, sn_s = [], [], [], [], [], []
    for t in range(L):
        for b in range(n):
            wp = mrt_precoder(perf[b, t]); wd = mrt_precoder(pred[b, t]); wn = mrt_precoder(Hn[b, t])
            nv = noise_calibration(Ht[b, t], wp, snr_lin)
            sinr_p = mrc_sinr(Ht[b, t], wp, nv); sinr_d = mrc_sinr(Ht[b, t], wd, nv); sinr_n = mrc_sinr(Ht[b, t], wn, nv)
            sp_s.extend(np.log2(1+sinr_p)); sd_s.extend(np.log2(1+sinr_d)); sn_s.extend(np.log2(1+sinr_n))
            nb = args.n_bits // K
            for k in range(K):
                tx = np.random.randint(0, 2, nb)
                bp.append(ber(tx, qam_mod(tx, mo)*np.sqrt(sinr_p[k]*nv), mo, nv))
                bd.append(ber(tx, qam_mod(tx, mo)*np.sqrt(sinr_d[k]*nv), mo, nv))
                bn.append(ber(tx, qam_mod(tx, mo)*np.sqrt(sinr_n[k]*nv), mo, nv))

    r = {'speed': (sp+1)*10, 'bp': np.mean(bp), 'bd': np.mean(bd), 'bn': np.mean(bn),
         'sp': np.mean(sp_s)*K, 'sd': np.mean(sd_s)*K, 'sn': np.mean(sn_s)*K}
    results.append(r)
    print(f"  {(sp+1)*10:>3d}km/h | P:BER={r['bp']:.2e} SE={r['sp']:.3f} | DS:BER={r['bd']:.2e} SE={r['sd']:.3f} | NP:BER={r['bn']:.2e} SE={r['sn']:.3f}")

os.makedirs('result', exist_ok=True)
csv = f'result/ber_{args.mode}_{args.scenario}_{args.modulation}_SNR{args.snr_dl}.csv'
with open(csv, 'w') as f:
    f.write('Speed,Perfect_BER,DS_BER,NoPred_BER,Perfect_SE,DS_SE,NoPred_SE\n')
    for r in results:
        f.write(f"{r['speed']},{r['bp']:.6e},{r['bd']:.6e},{r['bn']:.6e},{r['sp']:.6f},{r['sd']:.6f},{r['sn']:.6f}\n")
print(f'CSV → {csv}')
