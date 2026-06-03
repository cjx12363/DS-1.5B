'''
evaluate_rate.py -- Achievable Rate Evaluation for LLM4CP-DS
Loads trained TDD/FDD models, predicts CSI, computes per-subcarrier
achievable rates with ZF precoding.
Plots: Perfect CSI / Predicted CSI / No-Prediction.

Usage:
  python evaluate_rate.py --mode tdd --scenario UMa
  python evaluate_rate.py --mode fdd --scenario UMi
'''
import argparse, os, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import rearrange
import h5py
from models.LLM4CP import Model
import gc

def _load_mat(h5path, key):
    with h5py.File(h5path, 'r') as f:
        d = f[key][:]
        if d.dtype.names:
            d = (d['real'] + 1j * d['imag']).transpose()
        else:
            d = d.transpose()
    return d

def compute_zf_rate(H_pred_c, H_true_c, H_last_c, snr_db=10):
    B, K, Nt = H_pred_c.shape
    fro2 = torch.norm(H_true_c, p=2, dim=-1) ** 2
    noise_var = (fro2 / Nt) * (10 ** (-snr_db / 10))
    w_pred = H_pred_c.conj() / (torch.norm(H_pred_c, dim=-1, keepdim=True) + 1e-12)
    w_true = H_true_c.conj() / (torch.norm(H_true_c, dim=-1, keepdim=True) + 1e-12)
    w_last = H_last_c.conj() / (torch.norm(H_last_c, dim=-1, keepdim=True) + 1e-12)
    gain_true   = torch.abs(torch.sum(H_true_c * w_true, dim=-1)) ** 2
    gain_pred   = torch.abs(torch.sum(H_true_c * w_pred, dim=-1)) ** 2
    gain_nopred = torch.abs(torch.sum(H_true_c * w_last, dim=-1)) ** 2
    rate_true   = torch.log2(1.0 + gain_true   / (noise_var + 1e-12))
    rate_pred   = torch.log2(1.0 + gain_pred   / (noise_var + 1e-12))
    rate_nopred = torch.log2(1.0 + gain_nopred / (noise_var + 1e-12))
    return rate_true, rate_pred, rate_nopred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',     type=str, default='tdd', choices=['tdd', 'fdd'])
    parser.add_argument('--scenario', type=str, default='UMa', choices=['UMa', 'UMi'])
    parser.add_argument('--snr',      type=float, default=10.0)
    parser.add_argument('--bs',       type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  Mode: {args.mode}  Scenario: {args.scenario}  SNR: {args.snr} dB')

    prev_len, pred_len = 16, 4
    K, Nt, Nr = 48, 16, 4

    prev_path = f'./data/test/{args.scenario}_H_U_his_test.mat'
    pred_path = f'./data/test/{args.scenario}_H_D_pre_test.mat' if args.mode == 'fdd' else f'./data/test/{args.scenario}_H_U_pre_test.mat'

    his_raw = _load_mat(prev_path, 'H_U_his')
    pre_raw = _load_mat(pred_path, 'H_D_pre' if args.mode == 'fdd' else 'H_U_pre')
    print(f'History: {his_raw.shape}  Future: {pre_raw.shape}')

    ckpt = './Weights/LLM4CP_DS.pth' if args.mode == 'tdd' else './Weights/LLM4CP_DS_FDD.pth'
    model = Model(llm_type='deepseek-1.5b', use_kd=False, use_lora=True,
                  d_ff=1536, d_model=1536, pred_len=pred_len, prev_len=prev_len,
                  K=K, UQh=4, UQv=4, BQh=1, BQv=1, use_gpu=1, gpu_id=0).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f'Model: {ckpt}')

    speeds = list(range(10))
    rate_true_all, rate_pred_all, rate_nopred_all = [], [], []

    with torch.no_grad():
        for sp in speeds:
            his_s = his_raw[[sp]].mean(axis=6)
            pre_s = pre_raw[[sp]].mean(axis=6)
            his_s = rearrange(his_s, 'v b l k n m c -> (v b) l (k n m c)')
            pre_s = rearrange(pre_s, 'v b l k n m c -> (v b) l (k n m c)')
            samples = his_s.shape[0]
            std_val = np.sqrt(np.std(np.abs(his_s)**2) + 1e-12)
            his_n, pre_n = his_s/std_val, pre_s/std_val

            # convert to real [B, T, 1536*2=3072]
            his_real = np.zeros((samples, prev_len, his_n.shape[-1]*2), dtype=np.float32)
            his_real[:,:,0::2] = his_n.real.astype(np.float32)
            his_real[:,:,1::2] = his_n.imag.astype(np.float32)

            preds = []
            for start in range(0, samples, args.bs):
                end = min(start+args.bs, samples)
                inp = torch.tensor(his_real[start:end]).to(device)
                out = model(inp, None, None, None)
                preds.append(out.cpu().numpy())
            pred_real = np.concatenate(preds, axis=0) * std_val

            # reshape to [samples, 4, 48, 16] complex
            pred_c = pred_real.reshape(samples, pred_len, -1, 2)
            pred_complex = pred_c[..., 0] + 1j * pred_c[..., 1]
            pred_complex = pred_complex.reshape(samples, pred_len, K, Nt)

            pre_c = pre_n.reshape(samples, pred_len, K, Nt, 2)
            pre_complex = (pre_c[...,0] + 1j*pre_c[...,1]) * std_val

            his_c = his_n.reshape(samples, prev_len, K, Nt, 2)
            his_complex = (his_c[...,0] + 1j*his_c[...,1]) * std_val
            last_his = his_complex[:, -1:, :, :].repeat(pred_len, axis=1)

            for t in range(pred_len):
                hp = torch.tensor(pred_complex[:,t,:,:], dtype=torch.complex64)
                ht = torch.tensor(pre_complex[:,t,:,:], dtype=torch.complex64)
                hl = torch.tensor(last_his[:,t,:,:], dtype=torch.complex64)
                rt, rp, rn = compute_zf_rate(hp, ht, hl, snr_db=args.snr)
                rate_true_all.append(rt.numpy())
                rate_pred_all.append(rp.numpy())
                rate_nopred_all.append(rn.numpy())
            print(f'  speed {sp+1}/10  ({(sp+1)*10} km/h)  done')

    rate_true_all   = np.concatenate(rate_true_all, axis=0)
    rate_pred_all   = np.concatenate(rate_pred_all, axis=0)
    rate_nopred_all = np.concatenate(rate_nopred_all, axis=0)
    rate_true_k   = np.mean(rate_true_all, axis=0)
    rate_pred_k   = np.mean(rate_pred_all, axis=0)
    rate_nopred_k = np.mean(rate_nopred_all, axis=0)
    rt_mean, rp_mean, rn_mean = np.mean(rate_true_k), np.mean(rate_pred_k), np.mean(rate_nopred_k)

    print(f'\n=== {args.mode.upper()} / {args.scenario} ===')
    print(f'  Perfect CSI:    {rt_mean:.4f} bits/s/Hz')
    print(f'  Predicted CSI:  {rp_mean:.4f} bits/s/Hz')
    print(f'  No-prediction:  {rn_mean:.4f} bits/s/Hz')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(np.arange(K), rate_true_k,   'g-',  lw=2, label=f'Perfect CSI  ({rt_mean:.2f})')
    ax1.plot(np.arange(K), rate_pred_k,   'b--', lw=2, label=f'Predicted CSI ({rp_mean:.2f})')
    ax1.plot(np.arange(K), rate_nopred_k, 'r:',  lw=2, label=f'No-Prediction ({rn_mean:.2f})')
    ax1.set_xlabel('Subcarrier'); ax1.set_ylabel('Rate (bits/s/Hz)')
    ax1.set_title(f'Per-Subcarrier Rate ({args.mode.upper()}/{args.scenario})')
    ax1.legend(); ax1.grid(alpha=0.3)
    means = [rt_mean, rp_mean, rn_mean]
    labels = ['Perfect CSI', 'Predicted CSI', 'No-Prediction']
    colors = ['green', 'steelblue', 'tomato']
    bars = ax2.bar(labels, means, color=colors, edgecolor='black', lw=0.8)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean rate (bits/s/Hz)')
    ax2.set_title(f'Average Rate ({args.mode.upper()}/{args.scenario})')
    ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    png = f'rate_{args.mode}_{args.scenario}.png'
    plt.savefig(png, dpi=150); plt.close()
    print(f'Figure -> {png}')

    csv = f'rate_{args.mode}_{args.scenario}.csv'
    np.savetxt(csv, np.stack([rate_true_k, rate_pred_k, rate_nopred_k], axis=0).T,
               delimiter=',', header='perfect,predicted,nopred', comments='')
    print(f'CSV -> {csv}')
    print('Done.')
