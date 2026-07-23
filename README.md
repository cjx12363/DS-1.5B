# DS-1.5B: Lightweight LLM-based End-to-End CSI Prediction for MIMO-OFDM Systems

A lightweight LoRA-fine-tuned [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) model for end-to-end CSI prediction in MIMO-OFDM systems. By treating CSI as a temporal sequence and leveraging pre-trained LLM capabilities, DS-1.5B achieves state-of-the-art prediction accuracy with only **0.15M** trainable LoRA parameters.

## System Setup

| Parameter | Value |
|-----------|-------|
| Channel Model | QuaDRiGa (3GPP 38.901) |
| Scenario | UMa NLOS |
| Frequency | 2.4 GHz |
| Subcarriers | 48 (180 kHz spacing) |
| BS Antennas | 4×4 UPA (Nt = 16) |
| UE Antennas | 1×4 ULA (Nr = 4) |
| Prediction | 16 history steps → 4 future steps |
| Speed Range | 10–100 km/h |
| Modes | TDD / FDD |

## Installation

```bash
pip install -r requirements.txt
```

## Model Weights

Download pre-trained weights from [Releases](https://github.com/cjx12363/DS-1.5B/releases) and place in `./Weights/`:

```
Weights/
├── model.pth          # TDD model
└── U2D_model.pth      # FDD model
```

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
# Spectral efficiency (MRC receiver)
python eval.py --task se --mode tdd --scenario UMa

# Spectral efficiency (EGC receiver)
python eval.py --task se --mode tdd --combining egc

# BER evaluation
python eval.py --task ber --mode tdd --modulation QPSK

# Run all tasks
python eval.py --task all
```

**Arguments:**

| Flag | Options | Description |
|------|---------|-------------|
| `--task` | `se`, `ber`, `rate`, `nmse`, `all` | Evaluation task |
| `--mode` | `tdd`, `fdd` | Duplex mode |
| `--scenario` | `UMa`, `UMi` | Channel scenario |
| `--combining` | `mrc`, `egc` | Receiver combining method |
| `--modulation` | `QPSK`, `16QAM`, `64QAM` | Modulation (BER task) |
| `--snr_dl` | integer | Downlink SNR in dB (default: 10) |

## Key Results (TDD, SNR = 10 dB)

| Model | MRC SE | EGC SE |
|-------|--------|--------|
| **DS-1.5B** | **8.796** | **7.464** |
| PAD | 8.738 | 7.415 |
| CNN | 8.755 | 7.429 |
| GRU | 8.565 | 7.266 |
| LSTM | 8.555 | 7.258 |
| RNN | 8.388 | 7.115 |
| No Prediction | 7.715 | 6.539 |

*Perfect CSI upper bound: MRC 9.32 / EGC 7.91 bit/s/Hz*

## Citation

```bibtex
@article{ds15b2026,
  title={Lightweight LLM-based End-to-End CSI Prediction for MIMO-OFDM Systems},
  author={},
  journal={},
  year={2026}
}
```

## License

## License

MIT License — see [LICENSE](LICENSE) for details.
