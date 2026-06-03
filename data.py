import torch.utils.data as data
import torch
import numpy as np
import h5py
from einops import rearrange
from numpy import random
import gc

def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise

def _load_mat_chunked(h5path, key, train_per=0.9, valid_per=0.1, is_train=1):
    with h5py.File(h5path, 'r') as f:
        d = f[key]
        raw_shape = d.shape  # MATLAB dims: (2,4,4,4,48,16,n,v) or similar
        # After transpose, shape will be (v, n, L, K, Nt_h, Nt_v, Nr, 2)
        V = raw_shape[-1]  # speed dim
        N = raw_shape[-2]  # samples per speed
        n_train = int(train_per * N)
        n_valid = int(valid_per * N)

        chunks = []
        for v_idx in range(V):
            # Read one speed slice
            s = d[..., :, v_idx:v_idx+1]  # (2,4,4,4,48,16,N,1)
            if s.dtype.names:
                s = (s['real'] + 1j * s['imag']).transpose()
            else:
                s = s.transpose()
            s = s.squeeze(0)  # remove speed dim: (N, L, K, Nt_h, Nt_v, Nr, 2)

            if is_train:
                s = s[:n_train, ...]
            else:
                s = s[n_train:n_train+n_valid, ...]

            # UE merge (mean over Nr dim, axis=5 after squeeze)
            s = s.mean(axis=5)  # -> (n, L, K, Nt_h, Nt_v, 2)
            # Flatten: (n, L, K*Nt_h*Nt_v*2)
            s = rearrange(s, 'n l k a b c -> n l (k a b c)')
            chunks.append(s)

        result = np.concatenate(chunks, axis=0)
    return result

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path_r, file_path_t, is_train=1, ir=1, SNR=15, is_U2D=0, is_few=0,
                 train_per=0.9, valid_per=0.1):
        super(Dataset_Pro, self).__init__()
        self.SNR = SNR
        self.ir = ir

        print(f'[load] his: {file_path_r}', flush=True)
        H_his = _load_mat_chunked(file_path_r, 'H_U_his', train_per, valid_per, is_train)
        print(f'  his shape: {H_his.shape}', flush=True)
        gc.collect()

        key_t = 'H_D_pre' if is_U2D else 'H_U_pre'
        print(f'[load] pre: {file_path_t} key={key_t}', flush=True)
        H_pre = _load_mat_chunked(file_path_t, key_t, train_per, valid_per, is_train)
        print(f'  pre shape: {H_pre.shape}', flush=True)
        gc.collect()

        self.pred_len = H_pre.shape[1]
        self.prev_len = H_his.shape[1]

        # shuffle
        B = H_pre.shape[0]
        dt_all = np.concatenate((H_his, H_pre), axis=1)
        del H_his, H_pre; gc.collect()
        np.random.shuffle(dt_all)
        H_his = dt_all[:, :self.prev_len, ...]
        H_pre = dt_all[:, -self.pred_len:, ...]
        del dt_all; gc.collect()

        # add noise
        for i in range(B):
            H_his[i, ...] = noise(H_his[i, ...], random.rand() * 15 + 5.0)
            H_pre[i, ...] = noise(H_pre[i, ...], random.rand() * 15 + 5.0)

        # normalise
        std = np.sqrt(np.std(np.abs(H_his) ** 2) + 1e-12)
        H_his = H_his / std
        H_pre = H_pre / std

        # complex -> real interleaved tensor
        M, T, Mul = H_his.shape
        H_his_r = np.zeros((M, T, Mul * 2), dtype=np.float32)
        H_his_r[:, :, 0::2] = H_his.real.astype(np.float32)
        H_his_r[:, :, 1::2] = H_his.imag.astype(np.float32)
        H_his = torch.tensor(H_his_r, dtype=torch.float32)
        del H_his_r; gc.collect()

        P, Tp, Mp = H_pre.shape
        H_pre_r = np.zeros((P, Tp, Mp * 2), dtype=np.float32)
        H_pre_r[:, :, 0::2] = H_pre.real.astype(np.float32)
        H_pre_r[:, :, 1::2] = H_pre.imag.astype(np.float32)
        H_pre = torch.tensor(H_pre_r, dtype=torch.float32)
        del H_pre_r; gc.collect()

        if is_few == 1:
            H_pre = H_pre[::10, ...]
            H_his = H_his[::10, ...]

        self.pred = H_pre
        self.prev = H_his
        print(f'[done] prev={self.prev.shape} pred={self.pred.shape}', flush=True)

    def __getitem__(self, index):
        return self.pred[index, :].float(), self.prev[index, :].float()

    def __len__(self):
        return self.prev.shape[0]
