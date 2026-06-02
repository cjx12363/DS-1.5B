# AGENTS.md

## Network
- HuggingFace is blocked directly. Use Clash proxy at http://127.0.0.1:7897:
  `powershell
  $env:HTTP_PROXY='http://127.0.0.1:7897'; $env:HTTPS_PROXY='http://127.0.0.1:7897'
  `
- If proxy fails for large files, fall back to mirror: $env:HF_ENDPOINT='https://hf-mirror.com'

## Environment
- Conda env: llm4cp (Python 3.10, PyTorch 2.0 CUDA 11.8)
- GPU: RTX 4060 Ti (8.6 GB VRAM)
- Activate: conda activate llm4cp

## Models
- DeepSeek-1.5B: E:\models\DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-7B: E:\models\DeepSeek-R1-Distill-Qwen-7B

## Git
- Remote: https://github.com/cjx12363/LLM4CP-DS
- Proxy configured globally: git config --global http.proxy http://127.0.0.1:7897
- Do NOT commit: .mat, .pth, .docx, __pycache__, .idea/, venv/, Weights/

## Data
- Training data requires QuaDRiGa (MATLAB) to generate .mat files
- TDD: H_U_his_train.mat, H_U_pre_train.mat
- FDD: H_D_pre_train.mat (additional)
