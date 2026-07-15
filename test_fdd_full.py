"""FDD 测试: NMSE + MIMO MRC SE"""
import argparse, os, time, numpy as np, torch
from eval_utils import *
from models.LLM4CP import Model

def nmse(x_hat, x):
    return (torch.sum((x-x_hat)**2)/torch.sum(x**2)).item()

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str, default='UMa', choices=['UMa', 'UMi'])
parser.add_argument('--snr_dl', type=int, default=10)
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
P, L, K, Nt = 16, 4, 48, 16

his_raw = load_mat(f'./data/test/{args.scenario}_H_U_his_test.mat', 'H_U_his')
pre_raw = load_mat(f'./data/test/{args.scenario}_H_D_pre_test.mat', 'H_D_pre')

model = Model(llm_type='deepseek-1.5b', use_kd=False, use_lora=True, d_ff=1536, d_model=1536,
              pred_len=L, prev_len=P, K=K, UQh=4, UQv=4, BQh=1, BQv=1, use_gpu=1, gpu_id=0).to(device)
model.load_state_dict(torch.load('./Weights/U2D_LLM4CP_DS.pth', map_location=device)); model.eval()
print(f'FDD {args.scenario} SNR={args.snr_dl}dB | MRT+MRC')

nmse_all, se_all = [], []
for sp in range(10):
    hs, ps = his_raw[[sp]], pre_raw[[sp]]
    n = hs.shape[1]
    hc, pc = collapse_for_model(hs), collapse_for_model(ps)
    std = np.sqrt(np.std(np.abs(hc)**2) + 1e-12)

    inp, tgt = to_real(hc/std), to_real(pc/std)
    Ht = full_mimo_channel(ps)
    Hn = full_mimo_channel(hs)[:, -1:, :, :, :].repeat(L, axis=1)
    bs = 64; cycles = n // bs
    snr_lin = 10**(args.snr_dl/10)

    nmse_s, se_ps, se_ds, se_ns = [], [], [], []
    with torch.no_grad():
        for c in range(cycles):
            ib = torch.tensor(inp[c*bs:(c+1)*bs]).to(device)
            tb = torch.tensor(tgt[c*bs:(c+1)*bs]).to(device)
            out = model(ib, None, None, None)
            nmse_s.append(nmse(out, tb))

            pred = to_complex(out.cpu().numpy()*std, bs, L, K, Nt)
            perf = to_complex(tb.cpu().numpy()*std, bs, L, K, Nt)
            b0 = c*bs; b1 = (c+1)*bs
            for t in range(L):
                for b in range(bs):
                    wp = mrt_precoder(perf[b, t]); wd = mrt_precoder(pred[b, t]); wn = mrt_precoder(Hn[b0+b, t])
                    nv = noise_calibration(Ht[b0+b, t], wp, snr_lin)
                    se_ps.append(mrc_se(Ht[b0+b, t], wp, nv))
                    se_ds.append(mrc_se(Ht[b0+b, t], wd, nv))
                    se_ns.append(mrc_se(Ht[b0+b, t], wn, nv))

    nmse_v = np.nanmean(nmse_s)
    se_p = np.nanmean(se_ps); se_d = np.nanmean(se_ds); se_n = np.nanmean(se_ns)
    nmse_all.append(nmse_v); se_all.append(se_d/se_p)
    print(f'speed {sp}: NMSE={nmse_v:.6f} | P={se_p:.3f} D={se_d:.3f} ({se_d/se_p*100:.1f}%) N={se_n:.3f} ({se_n/se_p*100:.1f}%)')

os.makedirs('result', exist_ok=True)
with open(time.strftime('result/%Y%m%d_%H%M%S')+'_nmse_fdd.csv', 'w') as f:
    f.write(','.join(map(str, nmse_all))+'\n')
    f.write(','.join(map(str, se_all))+'\n')
print('Done.')
