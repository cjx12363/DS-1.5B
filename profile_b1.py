import sys; sys.path.insert(0, r'E:\cjx12363\LLM4CP-DS')
import torch, time
DEVICE = 'cuda:0'

class RNNetc:
    pass

# Quick batch=1 rerun for baselines
import numpy as np
from einops import rearrange

def LoadBatch_ofdm(H, num=32):
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) -> (b a) t k', a=num)
    H_real = np.zeros([B*num, T, mul//num, 2])
    H_real[:,:,:,0] = H.real; H_real[:,:,:,1] = H.imag
    return torch.tensor(H_real.reshape([B*num, T, mul//num*2]), dtype=torch.float32)

FEATURES, PREV_LEN = 96, 16

for name, fname in [('RNN','U2U_rnn.pth'),('GRU','U2U_gru.pth'),('LSTM','U2U_lstm.pth'),('CNN','U2U_cnn.pth')]:
    try:
        model = torch.load(f'Weights/full_shot_tdd/{fname}', map_location=DEVICE)
        model.eval().to(DEVICE)
        inp = torch.randn(1, PREV_LEN, FEATURES).to(DEVICE)
        cls = type(model).__name__
        with torch.no_grad():
            for _ in range(3):
                if cls in ('RNN','GRU','LSTM'):
                    _ = model(inp, 4, DEVICE)
                else:
                    try: _ = model(inp, None, None, None)
                    except: _ = model(inp)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(100):
                if cls in ('RNN','GRU','LSTM'):
                    _ = model(inp, 4, DEVICE)
                else:
                    try: _ = model(inp, None, None, None)
                    except: _ = model(inp)
            torch.cuda.synchronize()
            t = (time.time()-t0)/100*1000
        print(f'{name}: {t:.2f} ms (batch=1)')
    except Exception as e:
        print(f'{name}: failed ({e})')

print('DS-1.5B: ~85 ms (batch=1, estimated from batch=256 per-sample)')
