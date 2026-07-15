import sys; sys.path.insert(0, r"E:\cjx12363\LLM4CP-DS")
import torch, numpy as np, hdf5storage, os, time
from einops import rearrange

def LoadBatch_ofdm(H, num=32):
    B, T, mul = H.shape
    H = rearrange(H, "b t (k a) -> (b a) t k", a=num)
    H_real = np.zeros([B*num, T, mul//num, 2])
    H_real[:,:,:,0] = H.real; H_real[:,:,:,1] = H.imag
    return torch.tensor(H_real.reshape([B*num, T, mul//num*2]), dtype=torch.float32)

def run_test(mode="tdd", shot="full", device="cpu"):
    base = r"E:\cjx12363\LLM4CP-DS"
    prev_path = f"{base}/data/test/UMa_H_U_his_test.mat"
    pred_path = f"{base}/data/test/UMa_H_U_pre_test.mat"
    
    prev = hdf5storage.loadmat(prev_path)["H_U_his"]
    pred = hdf5storage.loadmat(pred_path)["H_U_pre"]
    prev = prev.mean(axis=6); pred = pred.mean(axis=6)
    prev = rearrange(prev, "v b l k n m c -> (v b) l (k n m c)")
    pred = rearrange(pred, "v b l k n m c -> (v b) l (k n m c)")
    std = np.sqrt(np.std(np.abs(prev)**2))
    prev, pred = prev/std, pred/std
    pv = LoadBatch_ofdm(prev, 32).to(device)
    pd = LoadBatch_ofdm(pred, 32).to(device)
    
    wdir = f"Weights/{shot}_shot_{mode}"
    name_map = {"tdd": "U2U", "fdd": "U2D"}
    prefix = name_map[mode]
    
    methods = {
        "No-Prediction": None,
        "CNN": f"{wdir}/{prefix}_cnn.pth",
        "RNN": f"{wdir}/{prefix}_rnn.pth",
        "GRU": f"{wdir}/{prefix}_gru.pth",
        "LSTM": f"{wdir}/{prefix}_lstm.pth",
        "Transformer": f"{wdir}/{prefix}_trans.pth",
        "LLM4CP (GPT-2)": f"{wdir}/{prefix}_LLM4CP.pth",
    }
    
    results = {}
    bs = 64
    for name, wpath in methods.items():
        print(f"[{name}]", end=" ", flush=True)
        if name == "No-Prediction":
            out = pv.view(1000, 32, 16, 96)[:, :, -1:, :].reshape(32000, 1, 96).repeat(1, 4, 1)
        else:
            model = torch.load(wpath, map_location=device)
            model.eval()
            model.to(device)
            cls = type(model).__name__
            out_list = []
            with torch.no_grad():
                for i in range(0, len(pv), bs):
                    inp = pv[i:i+bs]
                    if cls in ("RNN", "GRU", "LSTM"):
                        o = model(inp, 4, device)
                    elif cls == "Autoencoder":
                        o = model(inp)
                    elif cls in ("InformerStack",):
                        o = model(inp, torch.zeros(inp.shape[0], 4, inp.shape[2]))
                    else:
                        try:
                            o = model(inp, None, None, None)
                        except:
                            o = model(inp)
                    out_list.append(o.cpu())
            out = torch.cat(out_list)
        mse = torch.nn.functional.mse_loss(out, pd.cpu()).item()
        results[name] = mse
        print(f"NMSE={mse:.6f}")
    
    print(f"\n=== {mode.upper()} {shot}-shot UMa ===")
    for n, v in results.items():
        print(f"  {n:20s}: {v:.6f}")
    return results

# Run TDD full-shot
print("="*50)
print("TDD Full-shot UMa")
print("="*50)
run_test("tdd", "full", "cpu")

print("\n" + "="*50)
print("FDD Full-shot UMa")
print("="*50)
run_test("fdd", "full", "cpu")
