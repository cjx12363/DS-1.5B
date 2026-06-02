"""
LLM4CP-DS TDD Test Script
"""
import time
import argparse
import torch
import numpy as np
from data import LoadBatch_ofdm_1, LoadBatch_ofdm_2, noise, Transform_TDD_FDD
from metrics import NMSELoss, SE_Loss
from einops import rearrange
import hdf5storage
import tqdm
from models.LLM4CP import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="UMa", choices=["UMa", "UMi"])
    args = parser.parse_args()
    device = torch.device('cuda:0')
    is_U2D = 0
    prev_path = "./data/test/" + args.scenario + "_H_U_his_test.mat"
    pred_path = "./data/test/" + args.scenario + "_H_U_pre_test.mat"
    model_path = './Weights/LLM4CP_DS.pth'

    prev_len, pred_len = 16, 4
    K, Nt, Nr = 48, 16, 4

    criterion = NMSELoss()
    criterion_se = SE_Loss(snr=10, device=device)
    NMSE_all = []
    SE_all = []

    test_data_prev_base = hdf5storage.loadmat(prev_path)['H_U_his_test']
    test_data_pred_base = hdf5storage.loadmat(pred_path)['H_U_pre_test']

    model = Model(llm_type='deepseek-1.5b', use_kd=False, use_lora=True,
                  d_ff=1536, d_model=1536, pred_len=pred_len, prev_len=prev_len,
                  K=K, UQh=4, UQv=4, BQh=1, BQv=1, use_gpu=1, gpu_id=0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'Model loaded from {model_path}')

    for speed in range(0, 10):
        test_loss_stack, test_se_stack, test_se0_stack = [], [], []
        test_data_prev = test_data_prev_base[[speed], ...]
        test_data_pred = test_data_pred_base[[speed], ...]
        test_data_prev = test_data_prev.mean(axis=6)  # merge UE Nr=4
        test_data_prev = rearrange(test_data_prev, 'v b l k n m c -> (v b c) (n m) l (k)')
        test_data_pred = test_data_pred.mean(axis=6)  # merge UE Nr=4
        test_data_pred = rearrange(test_data_pred, 'v b l k n m c -> (v b c) (n m) l (k)')
        std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
        test_data_prev = test_data_prev / std
        test_data_pred = test_data_pred / std

        prev_data = LoadBatch_ofdm_2(test_data_prev)
        pred_data = LoadBatch_ofdm_2(test_data_pred)
        lens = prev_data.shape[0]
        bs = 64
        cycles = lens // bs

        with torch.no_grad():
            for cyt in range(cycles):
                prev = prev_data[cyt*bs:(cyt+1)*bs].to(device)
                pred = pred_data[cyt*bs:(cyt+1)*bs].to(device)
                prev = rearrange(prev, 'b m l k -> (b m) l k')
                pred = rearrange(pred, 'b m l k -> (b m) l k')
                out = model(prev, None, None, None)
                loss = criterion(out, pred)
                out_r = rearrange(out, '(b m) l k -> b l (k m)', b=bs)
                pred_r = rearrange(pred, '(b m) l k -> b l (k m)', b=bs)
                se, se0 = criterion_se(h=Transform_TDD_FDD(out_r, Nt=4*4, Nr=1),
                                       h0=Transform_TDD_FDD(pred_r, Nt=4*4, Nr=1))
                test_loss_stack.append(loss.item())
                test_se_stack.append(se.item())
                test_se0_stack.append(se0.item())

        nmse_val = np.nanmean(np.array(test_loss_stack))
        se_ratio = np.nanmean(np.array(test_se_stack)) / np.nanmean(np.array(test_se0_stack))
        print(f'speed {speed}: NMSE={nmse_val:.6f} SE_ratio={se_ratio:.4f}')
        NMSE_all.append(nmse_val)
        SE_all.append(se_ratio)

    fout = open(time.strftime("%Y%m%d_%H%M%S") + "_nmse_tdd.csv", "w")
    fout.write(','.join(map(str, NMSE_all)) + '\n')
    fout.write(','.join(map(str, SE_all)) + '\n')
    fout.close()
    print('Done.')
