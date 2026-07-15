"""
一键运行全部评估 (SE + BER)
用法:
  python run_all_eval.py --mode tdd --scenario UMa
  python run_all_eval.py --mode fdd --scenario UMa
  python run_all_eval.py --mode all  --scenario UMa

生成:
  result/e2e_{mode}_{scenario}.csv         — SE (DS-1.5B, Perfect, NoPred)
  result/ber_{mode}_{scenario}_QPSK_SNR10.csv  — BER QPSK
  result/ber_{mode}_{scenario}_16QAM_SNR10.csv — BER 16QAM

环境要求: GPU (CUDA), PyTorch, 模型权重在 ./Weights/
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('result', exist_ok=True)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'tdd'
SCENARIO = sys.argv[2] if len(sys.argv) > 2 else 'UMa'

modes = ['tdd', 'fdd'] if MODE == 'all' else [MODE]

for mode in modes:
    print(f"\n{'='*60}")
    print(f"  {mode.upper()} {SCENARIO} — 端到端 SE 评估")
    print(f"{'='*60}")
    subprocess.run([
        sys.executable, 'end_to_end_eval.py',
        '--mode', mode, '--scenario', SCENARIO,
        '--snr_dl', '10', '--max_samples', '200'
    ], check=False)

    for mod in ['QPSK', '16QAM']:
        print(f"\n{'='*60}")
        print(f"  {mode.upper()} {SCENARIO} — BER 评估 ({mod})")
        print(f"{'='*60}")
        subprocess.run([
            sys.executable, 'ber_eval.py',
            '--mode', mode, '--scenario', SCENARIO,
            '--modulation', mod, '--snr_dl', '10',
            '--max_samples', '200'
        ], check=False)

print("\n" + "="*60)
print("  全部评估完成！结果保存在 result/ 目录")
print("="*60)
print("\n生成文件:")
for f in sorted(os.listdir('result')):
    print(f"  result/{f}")
