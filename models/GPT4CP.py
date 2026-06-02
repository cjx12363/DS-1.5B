import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from Embed import DataEmbedding

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def get_peft_model_wrapper(base_model, lora_r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        if target_modules is None:
            target_modules = ['q_proj', 'v_proj']
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias='none',
        )
        model = get_peft_model(base_model, lora_config)
        return model
    except ImportError:
        print('[WARN] peft not installed, skipping LoRA.')
        return base_model


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs1)
        output = channel_attn * rs1
        rs = torch.add(x, output)
        return rs


class Model(nn.Module):
    def __init__(self,
                 gpt_type='deepseek-1.5b',
                 teacher_type='deepseek-7b',
                 gpt_layers=6,
                 use_lora=True,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.1,
                 use_kd=True,
                 kd_temperature=4.0,
                 d_ff=1536,
                 d_model=1536,
                 pred_len=4, prev_len=16,
                 use_gpu=1, gpu_id=0,
                 teacher_gpu_id=None,
                 mlp=0,
                 res_layers=4, res_dim=64,
                 K=48, UQh=4, UQv=4, BQh=1, BQv=1,
                 patch_size=4, stride=1,
                 embed='timeF', freq='h', dropout=0.1):
        super(Model, self).__init__()
        self.device = torch.device('cuda:{}'.format(gpu_id))
        self.teacher_gpu_id = teacher_gpu_id if teacher_gpu_id is not None else gpu_id
        self.teacher_device = torch.device('cuda:{}'.format(self.teacher_gpu_id))
        self.mlp = mlp
        self.res_layers = res_layers
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_ff = d_ff
        self.d_model = d_model
        self.use_kd = use_kd
        self.kd_temperature = kd_temperature
        self.K = K
        self.UQh = UQh
        self.UQv = UQv
        self.BQh = BQh
        self.BQv = BQv
        self.Nt = UQh * UQv
        self.Nr = BQh * BQv
        self.enc_in = K * UQh * UQv * BQh * BQv
        self.c_out = K * UQh * UQv * BQh * BQv
        self.enc_embedding1 = DataEmbedding(2 * self.enc_in, self.d_model, embed, freq, dropout)
        if gpt_type.startswith('deepseek'):
            self._init_deepseek_student(gpt_type, gpt_layers, use_lora, lora_r, lora_alpha, lora_dropout)
        else:
            self._init_gpt2_student(gpt_type, gpt_layers)
        self.teacher = None
        if use_kd and teacher_type is not None:
            self._init_teacher(teacher_type)
        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)
        self.patch_layer_fre = nn.Linear(self.patch_size, self.patch_size)
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)
        self.out_layer_dim = nn.Linear(d_ff, self.c_out * 2)
        self.output_layer_time = nn.Sequential(nn.Linear(self.prev_len, self.pred_len))
        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        self.RB_f = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(Res_block(res_dim))
            self.RB_f.append(Res_block(res_dim))
        self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.RB_f.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

    def _init_deepseek_student(self, model_name, gpt_layers, use_lora, lora_r, lora_alpha, lora_dropout):
        model_map = {
            'deepseek-1.5b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
            'deepseek-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        }
        model_path = model_map.get(model_name, model_name)
        print(f'[Student] Loading {model_path} ...')
        self.llm = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32)
        if hasattr(self.llm, 'layers'):
            self.llm.layers = self.llm.layers[:gpt_layers]
        elif hasattr(self.llm, 'encoder') and hasattr(self.llm.encoder, 'layer'):
            self.llm.encoder.layer = self.llm.encoder.layer[:gpt_layers]
        self.gpt_dim = self.llm.config.hidden_size
        print(f'[Student] Hidden dim: {self.gpt_dim}, Layers: {gpt_layers}')
        for name, param in self.llm.named_parameters():
            if any(k in name for k in ['self_attn', 'attn', 'input_layernorm', 'norm']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if use_lora:
            self.llm = get_peft_model_wrapper(self.llm, lora_r=lora_r, lora_alpha=lora_alpha,
                                               lora_dropout=lora_dropout, target_modules=['q_proj', 'v_proj'])
            print(f'[Student] LoRA: r={lora_r}, alpha={lora_alpha}')
        if hasattr(self, 'device'):
            self.llm.to(device=self.device)

    def _init_gpt2_student(self, gpt_type, gpt_layers):
        gpt_configs = {
            'gpt2': ('gpt2', 768), 'gpt2-medium': ('gpt2-medium', 1024),
            'gpt2-large': ('gpt2-large', 1280), 'gpt2-xl': ('gpt2-xl', 1600),
        }
        model_id, hidden_dim = gpt_configs.get(gpt_type, gpt_configs['gpt2'])
        self.llm = GPT2Model.from_pretrained(model_id, output_attentions=True, output_hidden_states=True)
        self.llm.h = self.llm.h[:gpt_layers]
        self.gpt_dim = hidden_dim
        for name, param in self.llm.named_parameters():
            if any(k in name for k in ['ln', 'wpe', 'attn', 'c_attn']):
                param.requires_grad = True
            elif 'mlp' in name and self.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if hasattr(self, 'device'):
            self.llm.to(device=self.device)

    def _init_teacher(self, teacher_type):
        model_map = {
            'deepseek-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
            'deepseek-1.5b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        }
        model_path = model_map.get(teacher_type, teacher_type)
        print(f'[Teacher] Loading {model_path} ...')
        self.teacher = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32)
        self.teacher_dim = self.teacher.config.hidden_size
        print(f'[Teacher] Hidden dim: {self.teacher_dim}')
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        self.teacher.to(device=self.teacher_device)
        if self.teacher_dim != self.gpt_dim:
            self.teacher_proj = nn.Linear(self.teacher_dim, self.gpt_dim).to(self.teacher_device)
        else:
            self.teacher_proj = nn.Identity()

    def _preprocess(self, x_enc):
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / std
        B, L, enc_in_dim = x_enc.shape
        x_enc_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)
        x_enc_complex = torch.complex(x_enc_r[:, :, :, 0], x_enc_r[:, :, :, 1])
        x_enc_delay = torch.fft.ifft(x_enc_complex, dim=2)
        x_enc_delay = torch.cat([torch.real(x_enc_delay), torch.imag(x_enc_delay)], dim=2)
        x_enc_delay = x_enc_delay.reshape(B, L // self.patch_size, self.patch_size, enc_in_dim)
        x_enc_delay = self.patch_layer(x_enc_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_delay = x_enc_delay.reshape(B, L, enc_in_dim)
        x_enc_delay = rearrange(x_enc_delay, 'b l (k o) -> b o l k', o=2)
        x_enc_delay = self.RB_f(x_enc_delay)
        x_enc_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, enc_in_dim)
        x_enc_fre = self.patch_layer(x_enc_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_enc_fre = x_enc_fre.reshape(B, L, enc_in_dim)
        x_enc_fre = rearrange(x_enc_fre, 'b l (k o) -> b o l k', o=2)
        x_enc_fre = self.RB_e(x_enc_fre)
        x_enc = x_enc_fre + x_enc_delay
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)
        return x_enc, mean, std

    def _llm_forward(self, enc_out):
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        if enc_out.shape[-1] < self.gpt_dim:
            enc_out = F.pad(enc_out, (0, self.gpt_dim - enc_out.shape[-1]))
        llm = self.llm
        dec_out = llm(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = self.out_layer_dim(dec_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, _ = x_enc.shape
        x_enc_processed, mean, std = self._preprocess(x_enc)
        enc_out = self.enc_embedding1(x_enc_processed, x_mark_enc)
        dec_out = self._llm_forward(enc_out)
        dec_out = dec_out * std + mean
        result = dec_out[:, -self.pred_len:, :]
        if self.use_kd and self.training and self.teacher is not None:
            with torch.no_grad():
                enc_out_t = enc_out.to(self.teacher_device)
                if enc_out_t.shape[-1] < self.teacher_dim:
                    enc_out_t = F.pad(enc_out_t, (0, self.teacher_dim - enc_out_t.shape[-1]))
                teacher_out = self.teacher(inputs_embeds=enc_out_t).last_hidden_state
                teacher_out = teacher_out[:, :, :self.teacher_dim]
                teacher_out = self.teacher_proj(teacher_out)
                teacher_out = teacher_out[:, :, :self.d_ff]
                teacher_dec = self.out_layer_dim(teacher_out)
                teacher_dec = self.output_layer_time(teacher_dec.permute(0, 2, 1)).permute(0, 2, 1)
                teacher_dec = teacher_dec.to(self.device)
                teacher_result = teacher_dec * std + mean
            return result, teacher_result[:, -self.pred_len:, :]
        return result


if __name__ == '__main__':
    import torch
    print('=== Testing with GPT-2 (backward compat) ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(gpt_type='gpt2', d_ff=768, d_model=768,
                  UQh=1, UQv=1, BQh=1, BQv=1, use_kd=False).to(device)
    inputs = torch.rand(3, 16, 96).to(device)
    out = model(inputs, None, None, None)
    print(f'Input: {inputs.shape} -> Output: {out.shape}')
    total = sum([p.nelement() for p in model.parameters()])
    print(f'Params: {total/1e6:.3f}M')
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {total_learn/1e6:.3f}M')
