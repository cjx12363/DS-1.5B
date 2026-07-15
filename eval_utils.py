"""MIMO 评估公共工具：数据加载、MRC SE/BER 计算"""
import numpy as np
import h5py
from einops import rearrange


def load_mat(path, key):
    with h5py.File(path, 'r') as f:
        d = f[key][:]
        if d.dtype.names:
            d = (d['real'] + 1j * d['imag']).transpose()
        else:
            d = d.transpose()
    return d


def collapse_for_model(data):
    """data: (1, samples, time, K, 4, 4, 4, 2) → (samples, time, 1536) complex"""
    c = data.mean(axis=-2)  # collapse Nr=4
    c = rearrange(c, 'v b l k a_row a_col pol -> (v b) l (k a_row a_col pol)')
    return c


def to_real(c):
    """complex → real interleaved"""
    r = np.zeros((c.shape[0], c.shape[1], c.shape[-1] * 2), dtype=np.float32)
    r[:, :, 0::2] = c.real.astype(np.float32)
    r[:, :, 1::2] = c.imag.astype(np.float32)
    return r

def to_complex(r, B, T, K, Nt, Npol=2):
    """real → complex, reshape to (B, T, K, Nt, Npol)"""
    c = r.reshape(B, T, -1, 2)
    c = c[..., 0] + 1j * c[..., 1]
    return c.reshape(B, T, K, Nt, Npol)


def full_mimo_channel(data):
    """data: (1, samples, time, K, 4, 4, 4, 2) → (samples, time, K, Nt=16, Nr_eff=8) complex"""
    return rearrange(data, 'v b l k a_row a_col rx pol -> (v b) l k (a_row a_col) (rx pol)')


def mrt_precoder(h_collapsed):
    """collapsed channel (..., K, Nt, Npol) → MRT precoder (..., K, Nt), mean over pol"""
    h = h_collapsed.mean(axis=-1)
    return np.conj(h) / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-12)


def mrc_sinr(H_true, w, noise_var):
    """H_true: (K, Nt, Nr_eff), w: (K, Nt) → (K,) SINR linear"""
    K = H_true.shape[0]
    rx = np.array([np.sum(np.abs(H_true[k].T.conj() @ w[k]) ** 2) for k in range(K)])
    return rx / noise_var


def mrc_se(H_true, w, noise_var):
    return np.sum(np.log2(1 + mrc_sinr(H_true, w, noise_var)))


def noise_calibration(H_true, w, snr_lin):
    K = H_true.shape[0]
    gain = np.array([np.sum(np.abs(H_true[k].T.conj() @ w[k]) ** 2) for k in range(K)])
    return np.mean(gain) / snr_lin
