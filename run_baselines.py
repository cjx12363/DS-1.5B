import sys; sys.path.insert(0,r"E:\cjx12363\LLM4CP-DS")
import torch,numpy as np,hdf5storage;from einops import rearrange;from tqdm import tqdm
D="cuda:0" if torch.cuda.is_available() else "cpu";print(f"GPU: {D}")

def ld(fp,k):
    d=hdf5storage.loadmat(fp)[k];d=d.mean(axis=6)
    d=rearrange(d,"v b l k n m c -> (v b c) (n m) l (k)")
    s=np.sqrt(np.std(np.abs(d)**2));d/=s
    B,M,L,K=d.shape;o=np.zeros([B,M,L,K*2],dtype=np.float32)
    o[:,:,:,0::2]=d.real;o[:,:,:,1::2]=d.imag;return torch.tensor(o)

bs=r"E:\cjx12363\LLM4CP-DS\data\test"
pv=ld(f"{bs}/UMa_H_U_his_test.mat","H_U_his")
pd=ld(f"{bs}/UMa_H_U_pre_test.mat","H_U_pre")
n_sp,B_sp=10,pv.shape[0]//10
print(f"Data: prev={pv.shape} pred={pd.shape}")

for mode in["tdd"]:
    pf="U2U" if mode=="tdd" else "U2D";wd=f"Weights/full_shot_{mode}"
    mt=[("No-Pred",None),("CNN",f"{wd}/{pf}_cnn.pth"),("GRU",f"{wd}/{pf}_gru.pth"),
        ("LSTM",f"{wd}/{pf}_lstm.pth"),("LLM4CP",f"{wd}/{pf}_LLM4CP.pth")]
    print(f"\n{'='*50}\n  {mode.upper()}")
    print(f"{'Method':12s}"+"".join(f"   S{s} " for s in range(n_sp))+"    Avg")
    for name,wp in mt:
        model=torch.load(wp,map_location=D).eval() if wp else None
        cls=type(model).__name__ if model else ""
        vals=[]
        for s in tqdm(range(n_sp),desc=f"  {name:10s}",leave=False):
            pvs=pv[s*B_sp:(s+1)*B_sp];pds=pd[s*B_sp:(s+1)*B_sp]
            if name=="No-Pred":
                out=pvs[:,:,-1:,:].repeat(1,1,4,1)
            else:
                out_list=[]
                for i in range(0,len(pvs),16):
                    x=rearrange(pvs[i:i+16].to(D),"b m l k -> (b m) l k")
                    with torch.no_grad():
                        if cls in("RNN","GRU","LSTM"):o=model(x,4,D)
                        elif cls=="Autoencoder":o=model(x)
                        else:o=model(x,None,None,None)
                    out_list.append(o.cpu())
                out=torch.cat(out_list)
            of=rearrange(out,"b l k -> (b l) k")
            pf=rearrange(pds,"b m l k -> (b m l) k")
            vals.append(torch.nn.functional.mse_loss(of,pf).item())
        r=f"{name:12s}"+"".join(f" {v:.4f}" for v in vals)+f"  {np.mean(vals):.5f}"
        print(r)
print("\nDone!")
