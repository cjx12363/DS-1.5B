"""
Test script for author's pretrained LLM4CP (GPT-2) weights
Uses original data.py format and GPT4CP model
"""
import torch
import numpy as np
import hdf5storage
import sys
import os
from einops import rearrange
from metrics import NMSELoss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def LoadBatch_ofdm(H, num=32):
    B, T, mul = H.shape
    H = rearrange(H, 'b t (k a) -> (b a) t k', a=num)
    H_real = np.zeros([B * num, T, mul // num, 2])
    H_real[:, :, :, 0] = H.real
    H_real[:, :, :, 1] = H.imag
    H_real = H_real.reshape([B * num, T, mul // num * 2])
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


if __name__ == '__main__':
    from models.GPT4CP import Model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='tdd', choices=['tdd', 'fdd'])
    parser.add_argument('--shot', type=str, default='full', choices=['full', 'few'])
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if args.shot == 'full':
        weight_dir = f'Weights/full_shot_{args.mode}'
        weight_name = 'U2U_LLM4CP.pth' if args.mode == 'tdd' else 'U2D_LLM4CP.pth'
    else:
        weight_dir = f'Weights/few_shot_{args.mode}'
        weight_name = 'U2U_LLM4CP_few.pth' if args.mode == 'tdd' else 'U2D_LLM4CP_few.pth'

    weight_path = os.path.join(weight_dir, weight_name)
    print(f'Loading model from {weight_path}')
    model = torch.load(weight_path, map_location=args.device)
    model.to(args.device)
    model.eval()
    print(f'Loaded. Params: {sum(p.numel() for p in model.parameters()):,}')

    test_prev_path = 'data/test/UMa_H_U_his_test.mat'
    test_pred_path = 'data/test/UMa_H_U_pre_test.mat'
    test_prev = hdf5storage.loadmat(test_prev_path)['H_U_his']
    test_pred = hdf5storage.loadmat(test_pred_path)['H_U_pre']
    print(f'Raw: prev={test_prev.shape}, pred={test_pred.shape}')

    # UE merge + flatten
    test_prev = test_prev.mean(axis=5)
    test_pred = test_pred.mean(axis=5)
    test_prev = rearrange(test_prev, 'v b l k n m c -> (v b) l (k n m c)')
    test_pred = rearrange(test_pred, 'v b l k n m c -> (v b) l (k n m c)')
    std = np.sqrt(np.std(np.abs(test_prev) ** 2))
    test_prev = test_prev / std
    test_pred = test_pred / std
    prev_data = LoadBatch_ofdm(test_prev, num=32)
    pred_data = LoadBatch_ofdm(test_pred, num=32)
    print(f'Model input: prev={prev_data.shape}, pred={pred_data.shape}')

    criterion = NMSELoss()
    NMSE_all = []
    bs = 64
    with torch.no_grad():
        prev = prev_data.to(args.device)
        pred = pred_data.to(args.device)
        n_batches = prev.shape[0] // bs
        for i in range(n_batches):
            out = model(prev[i*bs:(i+1)*bs], None, None, None)
            loss = criterion(out, pred[i*bs:(i+1)*bs])
            NMSE_all.append(loss.item())
    avg_nmse = np.mean(NMSE_all)
    print(f'NMSE: {avg_nmse:.6f}')
