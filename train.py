import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
from models.GPT4CP import Model
import numpy as np
import shutil
import argparse
from metrics import NMSELoss, SE_Loss


def parse_args():
    parser = argparse.ArgumentParser(description='LLM4CP-DS Training')
    parser.add_argument('--mode', type=str, default='tdd', choices=['tdd', 'fdd'])
    parser.add_argument('--few_shot', type=int, default=0)
    parser.add_argument('--zero_shot', type=int, default=0)
    parser.add_argument('--gpt_type', type=str, default='deepseek-1.5b')
    parser.add_argument('--teacher_type', type=str, default='deepseek-7b')
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--use_lora', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--use_kd', type=int, default=1)
    parser.add_argument('--kd_temperature', type=float, default=4.0)
    parser.add_argument('--kd_alpha', type=float, default=0.5)
    parser.add_argument('--d_ff', type=int, default=1536)
    parser.add_argument('--d_model', type=int, default=1536)
    parser.add_argument('--pred_len', type=int, default=4)
    parser.add_argument('--prev_len', type=int, default=16)
    parser.add_argument('--K', type=int, default=48)
    parser.add_argument('--UQh', type=int, default=4)
    parser.add_argument('--UQv', type=int, default=4)
    parser.add_argument('--BQh', type=int, default=1)
    parser.add_argument('--BQv', type=int, default=1)
    parser.add_argument('--res_layers', type=int, default=4)
    parser.add_argument('--res_dim', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--teacher_gpu_id', type=int, default=None)
    parser.add_argument('--train_r_path', type=str, default='./H_U_his_train.mat')
    parser.add_argument('--train_t_path', type=str, default='./H_U_pre_train.mat')
    parser.add_argument('--train_t_path_fdd', type=str, default='./H_D_pre_train.mat')
    parser.add_argument('--save_dir', type=str, default='./Weights')
    parser.add_argument('--save_name', type=str, default='LLM4CP_DS.pth')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    if args.teacher_type.lower() == 'none':
        args.teacher_type = None
    return args

class KDLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_pred, teacher_pred, target):
        hard_loss = F.mse_loss(student_pred, target)
        soft_student = F.log_softmax(student_pred / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_pred / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        total_loss = hard_loss + self.alpha * soft_loss
        return total_loss, hard_loss, soft_loss

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f'[Save] Checkpoint -> {path}')

def train(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    best_loss = float('inf')
    is_U2D = 1 if args.mode == 'fdd' else 0
    train_t_path = args.train_t_path_fdd if is_U2D else args.train_t_path
    print(f'[Data] Mode: {args.mode.upper()} | Few-shot: {args.few_shot}')
    train_set = Dataset_Pro(args.train_r_path, train_t_path, is_train=1, is_U2D=is_U2D, is_few=args.few_shot)
    val_set = Dataset_Pro(args.train_r_path, train_t_path, is_train=0, is_U2D=is_U2D, is_few=0)
    train_loader = DataLoader(train_set, num_workers=0, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, num_workers=0, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    teacher_type = args.teacher_type
    if args.use_kd == 0:
        teacher_type = None
    model = Model(
        gpt_type=args.gpt_type, teacher_type=teacher_type, gpt_layers=args.gpt_layers,
        use_lora=bool(args.use_lora), lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        use_kd=bool(args.use_kd), kd_temperature=args.kd_temperature,
        d_ff=args.d_ff, d_model=args.d_model,
        pred_len=args.pred_len, prev_len=args.prev_len,
        use_gpu=1, gpu_id=args.gpu_id, teacher_gpu_id=args.teacher_gpu_id,
        K=args.K, UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv,
        res_layers=args.res_layers, res_dim=args.res_dim, patch_size=args.patch_size,
    ).to(device)
    if args.resume and os.path.exists(args.resume):
        print(f'[Resume] Loading {args.resume}')
        model.load_state_dict(torch.load(args.resume, map_location=device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0001)
    criterion_kd = KDLoss(temperature=args.kd_temperature, alpha=args.kd_alpha).to(device)
    save_path = os.path.join(args.save_dir, args.save_name)
    total = sum(p.nelement() for p in model.parameters())
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Model] Total: {total/1e6:.3f}M | Trainable: {learnable/1e6:.3f}M')
    print(f'[KD] enabled={args.use_kd} T={args.kd_temperature} alpha={args.kd_alpha}')
    print('=' * 60)
    for epoch in range(args.epochs):
        epoch_train_loss, epoch_hard_loss, epoch_soft_loss = [], [], []
        epoch_val_loss = []
        model.train()
        for iteration, batch in enumerate(train_loader, 1):
            pred_t = Variable(batch[0]).to(device)
            prev = Variable(batch[1]).to(device)
            optimizer.zero_grad()
            if args.use_kd and teacher_type is not None:
                pred_m, teacher_m = model(prev, None, None, None)
                loss, hard_loss, soft_loss = criterion_kd(pred_m, teacher_m, pred_t)
                epoch_hard_loss.append(hard_loss.item())
                epoch_soft_loss.append(soft_loss.item())
            else:
                pred_m = model(prev, None, None, None)
                loss = F.mse_loss(pred_m, pred_t)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        h_str = f' hard={np.nanmean(np.array(epoch_hard_loss)):.6f}' if epoch_hard_loss else ''
        s_str = f' soft={np.nanmean(np.array(epoch_soft_loss)):.6f}' if epoch_soft_loss else ''
        print(f'Epoch {epoch+1}/{args.epochs} | loss={t_loss:.6f}{h_str}{s_str}')
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                pred_t = Variable(batch[0]).to(device)
                prev = Variable(batch[1]).to(device)
                pred_m = model(prev, None, None, None)
                epoch_val_loss.append(F.mse_loss(pred_m, pred_t).item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        print(f'  val_loss={v_loss:.6f}')
        if v_loss < best_loss:
            best_loss = v_loss
            save_checkpoint(model, save_path)
    print('=' * 60)
    print(f'Done. Best val loss: {best_loss:.6f}')

if __name__ == '__main__':
    args = parse_args()
    train(args)
