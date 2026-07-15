"""MIMO 评估: SE / BER / Rate / NMSE，统一入口
用法:
  python eval.py --task se    --mode tdd --scenario UMa
  python eval.py --task ber   --mode tdd --modulation QPSK
  python eval.py --task rate  --mode fdd --scenario UMa
  python eval.py --task nmse  --mode tdd --scenario UMa
"""
import argparse, os, time, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from eval_utils import *
from models.model import Model


def qam_mod(bits, mo):
    M, nm = 2**mo, bits.shape[0]//mo
    bits = bits[:nm*mo].reshape(nm, mo)
    sm = int(np.sqrt(M))
    ri = sum(bits[:, i]<<i for i in range(mo//2))
    ii = sum(bits[:, i+mo//2]<<i for i in range(mo//2))
    return ((2*ri-sm+1) + 1j*(2*ii-sm+1)) / np.sqrt((M-1)*2/3)

def qam_demod(syms, mo):
    M, sm = 2**mo, int(np.sqrt(M))
    syms *= np.sqrt((M-1)*2/3)
    rv = np.clip(np.round((syms.real+sm-1)/2), 0, sm-1).astype(int)
    iv = np.clip(np.round((syms.imag+sm-1)/2), 0, sm-1).astype(int)
    bits = np.zeros(len(syms)*mo, dtype=int)
    for i in range(mo//2): bits[i::mo] = (rv>>i)&1
    for i in range(mo//2): bits[mo//2+i::mo] = (iv>>i)&1
    return bits

def ber(tx, rx_sym, mo, nv):
    n = np.sqrt(nv/2)*(np.random.randn(*rx_sym.shape)+1j*np.random.randn(*rx_sym.shape))
    return np.mean(tx != qam_demod(rx_sym+n, mo))


def load_model(device, mode, P, L, K):
    ckpt = f'./Weights/{"U2D_" if mode=="fdd" else ""}model.pth'
    m = Model(llm_type='deepseek-1.5b', use_kd=False, use_lora=True, d_ff=1536, d_model=1536,
              pred_len=L, prev_len=P, K=K, UQh=4, UQv=4, BQh=1, BQv=1, use_gpu=1, gpu_id=0).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    return m

def predict(model, his_real, std, device, bs, n, L, K, Nt):
    preds = []
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s+bs, n)
            preds.append(model(torch.tensor(his_real[s:e]).to(device), None, None, None).cpu().numpy())
    return to_complex(np.concatenate(preds, axis=0)*std, n, L, K, Nt)


def task_se(args, device):
    P, L, K, Nt = 16, 4, 48, 16
    tag = 'D' if args.mode == 'fdd' else 'U'
    his_raw = load_mat(f'./data/test/{args.scenario}_H_U_his_test.mat', 'H_U_his')
    pre_raw = load_mat(f'./data/test/{args.scenario}_H_{tag}_pre_test.mat', f'H_{tag}_pre')
    model = load_model(device, args.mode, P, L, K)
    print(f'SE | {args.mode.upper()} {args.scenario} SNR={args.snr_dl}dB | Nt=16 Nr_eff=8')

    results = []
    for sp in range(10):
        hs, ps = his_raw[[sp]], pre_raw[[sp]]
        n = min(args.max_samples, hs.shape[1])
        hc, pc = collapse_for_model(hs[:,:n]), collapse_for_model(ps[:,:n])
        std = np.sqrt(np.std(np.abs(hc)**2)+1e-12)
        pred = predict(model, to_real(hc/std), std, device, args.bs, n, L, K, Nt)
        perf = to_complex(to_real(pc/std), n, L, K, Nt)*std
        Ht = full_mimo_channel(ps[:,:n])
        Hn = full_mimo_channel(hs[:,:n])[:, -1:].repeat(L, axis=1)

        sl = 10**(args.snr_dl/10)
        sp_s, sd_s, sn_s = [], [], []
        for t in range(L):
            for b in range(n):
                wp = mrt_precoder(perf[b,t]); wd = mrt_precoder(pred[b,t]); wn = mrt_precoder(Hn[b,t])
                nv = noise_calibration(Ht[b,t], wp, sl)
                sp_s.append(mrc_se(Ht[b,t], wp, nv))
                sd_s.append(mrc_se(Ht[b,t], wd, nv))
                sn_s.append(mrc_se(Ht[b,t], wn, nv))
        r = {'speed':(sp+1)*10, 'p':np.mean(sp_s), 'd':np.mean(sd_s), 'n':np.mean(sn_s)}
        r['d%']=r['d']/r['p']*100; r['n%']=r['n']/r['p']*100
        results.append(r)
        print(f"  {(sp+1)*10:>3d}km/h | P={r['p']:.3f} D={r['d']:.3f} ({r['d%']:.1f}%) N={r['n']:.3f} ({r['n%']:.1f}%)")

    avg = {k:np.mean([r[k] for r in results]) for k in ['p','d','n','d%','n%']}
    print(f"  Avg | P={avg['p']:.4f} D={avg['d']:.4f} ({avg['d%']:.1f}%) N={avg['n']:.4f} ({avg['n%']:.1f}%)")
    os.makedirs('result', exist_ok=True)
    with open(f'result/se_{args.mode}_{args.scenario}_SNR{args.snr_dl}.csv','w') as f:
        f.write('Speed,Perfect,DS-1.5B,DS_pct,NoPred,NP_pct\n')
        for r in results: f.write(f"{r['speed']},{r['p']:.6f},{r['d']:.6f},{r['d%']:.2f},{r['n']:.6f},{r['n%']:.2f}\n")


def task_ber(args, device):
    mo = {'QPSK':2,'16QAM':4,'64QAM':6}[args.modulation]
    P, L, K, Nt = 16, 4, 48, 16
    tag = 'D' if args.mode == 'fdd' else 'U'
    his_raw = load_mat(f'./data/test/{args.scenario}_H_U_his_test.mat', 'H_U_his')
    pre_raw = load_mat(f'./data/test/{args.scenario}_H_{tag}_pre_test.mat', f'H_{tag}_pre')
    model = load_model(device, args.mode, P, L, K)
    print(f'BER | {args.mode.upper()} {args.scenario} {args.modulation} SNR={args.snr_dl}dB')

    results = []
    for sp in range(10):
        hs, ps = his_raw[[sp]], pre_raw[[sp]]
        n = min(args.max_samples, hs.shape[1])
        hc, pc = collapse_for_model(hs[:,:n]), collapse_for_model(ps[:,:n])
        std = np.sqrt(np.std(np.abs(hc)**2)+1e-12)
        pred = predict(model, to_real(hc/std), std, device, args.bs, n, L, K, Nt)
        perf = to_complex(to_real(pc/std), n, L, K, Nt)*std
        Ht = full_mimo_channel(ps[:,:n])
        Hn = full_mimo_channel(hs[:,:n])[:, -1:].repeat(L, axis=1)

        sl = 10**(args.snr_dl/10)
        bp, bd, bn, sp_s, sd_s, sn_s = [], [], [], [], [], []
        for t in range(L):
            for b in range(n):
                wp=mrt_precoder(perf[b,t]); wd=mrt_precoder(pred[b,t]); wn=mrt_precoder(Hn[b,t])
                nv=noise_calibration(Ht[b,t], wp, sl)
                sip=mrc_sinr(Ht[b,t], wp, nv); sid=mrc_sinr(Ht[b,t], wd, nv); sin_=mrc_sinr(Ht[b,t], wn, nv)
                sp_s.extend(np.log2(1+sip)); sd_s.extend(np.log2(1+sid)); sn_s.extend(np.log2(1+sin_))
                nb=args.n_bits//K
                for k in range(K):
                    tx=np.random.randint(0,2,nb)
                    bp.append(ber(tx, qam_mod(tx,mo)*np.sqrt(sip[k]*nv), mo, nv))
                    bd.append(ber(tx, qam_mod(tx,mo)*np.sqrt(sid[k]*nv), mo, nv))
                    bn.append(ber(tx, qam_mod(tx,mo)*np.sqrt(sin_[k]*nv), mo, nv))
        r = {'speed':(sp+1)*10, 'bp':np.mean(bp), 'bd':np.mean(bd), 'bn':np.mean(bn),
             'sp':np.mean(sp_s)*K, 'sd':np.mean(sd_s)*K, 'sn':np.mean(sn_s)*K}
        results.append(r)
        print(f"  {(sp+1)*10:>3d}km/h | P:BER={r['bp']:.2e} SE={r['sp']:.3f} | D:BER={r['bd']:.2e} SE={r['sd']:.3f}")

    os.makedirs('result', exist_ok=True)
    with open(f'result/ber_{args.mode}_{args.scenario}_{args.modulation}_SNR{args.snr_dl}.csv','w') as f:
        f.write('Speed,P_BER,D_BER,N_BER,P_SE,D_SE,N_SE\n')
        for r in results: f.write(f"{r['speed']},{r['bp']:.6e},{r['bd']:.6e},{r['bn']:.6e},{r['sp']:.6f},{r['sd']:.6f},{r['sn']:.6f}\n")


def task_rate(args, device):
    P, L, K, Nt = 16, 4, 48, 16
    tag = 'D' if args.mode == 'fdd' else 'U'
    his_raw = load_mat(f'./data/test/{args.scenario}_H_U_his_test.mat', 'H_U_his')
    pre_raw = load_mat(f'./data/test/{args.scenario}_H_{tag}_pre_test.mat', f'H_{tag}_pre')
    model = load_model(device, args.mode, P, L, K)
    print(f'Rate | {args.mode.upper()} {args.scenario} SNR={args.snr}dB')

    rp_all, rd_all, rn_all = [], [], []
    for sp in range(10):
        hs, ps = his_raw[[sp]], pre_raw[[sp]]
        n = hs.shape[1]
        hc, pc = collapse_for_model(hs), collapse_for_model(ps)
        std = np.sqrt(np.std(np.abs(hc)**2)+1e-12)
        pred = predict(model, to_real(hc/std), std, device, args.bs, n, L, K, Nt)
        perf = to_complex(to_real(pc/std), n, L, K, Nt)*std
        Ht = full_mimo_channel(ps)
        Hn = full_mimo_channel(hs)[:, -1:].repeat(L, axis=1)

        sl = 10**(args.snr/10)
        for t in range(L):
            wp=mrt_precoder(perf[:,t]); wd=mrt_precoder(pred[:,t]); wn=mrt_precoder(Hn[:,t])
            gp = np.mean([noise_calibration(Ht[b,t], wp[b], sl)*sl for b in range(n)])
            nv = gp/sl
            for b in range(n):
                rp_all.append(np.log2(1+mrc_sinr(Ht[b,t], wp[b], nv)))
                rd_all.append(np.log2(1+mrc_sinr(Ht[b,t], wd[b], nv)))
                rn_all.append(np.log2(1+mrc_sinr(Ht[b,t], wn[b], nv)))
        print(f'  speed {sp+1}/10')

    rp=np.mean(np.concatenate(rp_all),0); rd=np.mean(np.concatenate(rd_all),0); rn=np.mean(np.concatenate(rn_all),0)
    print(f'  Perfect={np.mean(rp):.4f}  Pred={np.mean(rd):.4f}  NoPred={np.mean(rn):.4f}')

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    ax1.plot(range(K),rp,'g-',lw=2,label=f'Perfect ({np.mean(rp):.2f})')
    ax1.plot(range(K),rd,'b--',lw=2,label=f'Predicted ({np.mean(rd):.2f})')
    ax1.plot(range(K),rn,'r:',lw=2,label=f'No-Pred ({np.mean(rn):.2f})')
    ax1.set_xlabel('Subcarrier'); ax1.set_ylabel('Rate (bits/s/Hz)')
    ax1.set_title(f'Per-Subcarrier Rate ({args.mode.upper()}/{args.scenario})'); ax1.legend(); ax1.grid(alpha=0.3)
    for bar,val,c in zip(['Perfect','Predicted','No-Pred'],[np.mean(rp),np.mean(rd),np.mean(rn)],['green','steelblue','tomato']):
        b=ax2.bar(bar,val,color=c,edgecolor='black',lw=0.8)
        ax2.text(b[0].get_x()+b[0].get_width()/2,val+0.02,f'{val:.3f}',ha='center',fontsize=11,fontweight='bold')
    ax2.set_ylabel('Mean rate'); ax2.set_title('Average Rate'); ax2.grid(alpha=0.3,axis='y')
    plt.tight_layout(); plt.savefig(f'rate_{args.mode}_{args.scenario}.png',dpi=150); plt.close()
    np.savetxt(f'rate_{args.mode}_{args.scenario}.csv', np.stack([rp,rd,rn],0).T, delimiter=',', header='perfect,predicted,nopred', comments='')


def task_nmse(args, device):
    P, L, K, Nt = 16, 4, 48, 16
    tag = 'D' if args.mode == 'fdd' else 'U'
    his_raw = load_mat(f'./data/test/{args.scenario}_H_U_his_test.mat', 'H_U_his')
    pre_raw = load_mat(f'./data/test/{args.scenario}_H_{tag}_pre_test.mat', f'H_{tag}_pre')
    model = load_model(device, args.mode, P, L, K)
    print(f'NMSE | {args.mode.upper()} {args.scenario} SNR={args.snr_dl}dB')

    nmse_all, se_all = [], []
    for sp in range(10):
        hs, ps = his_raw[[sp]], pre_raw[[sp]]
        n = hs.shape[1]
        hc, pc = collapse_for_model(hs), collapse_for_model(ps)
        std = np.sqrt(np.std(np.abs(hc)**2)+1e-12)
        inp, tgt = to_real(hc/std), to_real(pc/std)
        Ht = full_mimo_channel(ps)
        Hn = full_mimo_channel(hs)[:, -1:].repeat(L, axis=1)
        bs=64; cycles=n//bs
        sl=10**(args.snr_dl/10)

        ns, sp_s, sd_s, sn_s = [], [], [], []
        with torch.no_grad():
            for c in range(cycles):
                ib=torch.tensor(inp[c*bs:(c+1)*bs]).to(device); tb=torch.tensor(tgt[c*bs:(c+1)*bs]).to(device)
                out=model(ib,None,None,None)
                ns.append((torch.sum((out-tb)**2)/torch.sum(tb**2)).item())
                pred=to_complex(out.cpu().numpy()*std, bs, L, K, Nt)
                perf=to_complex(tb.cpu().numpy()*std, bs, L, K, Nt)
                for t in range(L):
                    for b in range(bs):
                        wp=mrt_precoder(perf[b,t]); wd=mrt_precoder(pred[b,t]); wn=mrt_precoder(Hn[c*bs+b,t])
                        nv=noise_calibration(Ht[c*bs+b,t], wp, sl)
                        sp_s.append(mrc_se(Ht[c*bs+b,t], wp, nv))
                        sd_s.append(mrc_se(Ht[c*bs+b,t], wd, nv))
                        sn_s.append(mrc_se(Ht[c*bs+b,t], wn, nv))
        nv=np.nanmean(ns); sp=np.nanmean(sp_s); sd=np.nanmean(sd_s); sn_=np.nanmean(sn_s)
        nmse_all.append(nv); se_all.append(sd/sp)
        print(f'  speed {sp}: NMSE={nv:.6f} | P={sp:.3f} D={sd:.3f} ({sd/sp*100:.1f}%) N={sn_:.3f}')

    os.makedirs('result', exist_ok=True)
    with open(time.strftime(f'result/%Y%m%d_%H%M%S_nmse_{args.mode}.csv'),'w') as f:
        f.write(','.join(map(str,nmse_all))+'\n'); f.write(','.join(map(str,se_all))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['se','ber','rate','nmse','all'])
    parser.add_argument('--mode', type=str, default='tdd', choices=['tdd','fdd'])
    parser.add_argument('--scenario', type=str, default='UMa', choices=['UMa','UMi'])
    parser.add_argument('--modulation', type=str, default='QPSK', choices=['QPSK','16QAM','64QAM'])
    parser.add_argument('--snr_dl', type=int, default=10)
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--n_bits', type=int, default=100000)
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tasks = ['se','ber','rate','nmse'] if args.task == 'all' else [args.task]
    mods = ['QPSK','16QAM'] if args.task in ('ber','all') else [args.modulation]
    modes = ['tdd','fdd'] if args.task == 'all' else [args.mode]

    for mode in modes:
        args.mode = mode
        for t in tasks:
            if t == 'ber':
                for mod in mods:
                    args.modulation = mod
                    task_ber(args, device)
            elif t == 'se': task_se(args, device)
            elif t == 'rate': task_rate(args, device)
            elif t == 'nmse': task_nmse(args, device)
    print('Done.')
