# SynthCloner: Synthesizer Preset Conversion via Factorized Codec with Disentangled Timbre and ADSR Control

<div align="center">

**Jeng-Yue Liu<sup>1, 2</sup>, Ting-Chao Hsu<sup>1</sup>, Yen-Tung Yeh<sup>1</sup>, Li Su<sup>2</sup>, Yi-Hsuan Yang<sup>1</sup>**

<sup>1</sup> National Taiwan University  
<sup>2</sup> Academia Sinica

[![Demo](https://img.shields.io/badge/Demo-Live%20Demo-blue)](https://buffett0323.github.io/synthcloner/)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/buffett0323/synthcloner)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2504.18157) -->

</div>

## Abstract

Electronic synthesizer sounds are controlled by presets, parameters settings that yield complex timbral characteristics and ADSR envelopes, making preset conversion particularly challenging. Recent approaches to timbre transfer often rely on spectral objectives or implicit style matching, offering limited control over envelope shaping. Moreover, public synthesizer datasets rarely provide diverse coverage of timbres and ADSR envelopes. To address these gaps, we present SynthCloner, a factorized codec model that disentangles audio into three attributes: ADSR envelope, timbre, and content. This separation enables expressive synthesizer preset conversion with independent control over these three attributes. Additionally, we introduce SynthCAT, a new synthesizer dataset with a task-specific rendering pipeline covering 250 timbres, 120 ADSR envelopes, and 100 MIDI sequences. Experiments show that SynthCloner outperforms baselines on both objective and subjective metrics, while enabling independent attribute control.

## Quick Start

### Prerequisites

- Python 3.10.13
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/buffett0323/synthcloner.git
   cd synthcloner
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pretrained models:**
   ```bash
   # TODO: Put model checkpoints
   # Download model checkpoints to ./checkpoints/
   # (Model links will be provided upon publication)
   ```

### Basic Usage

#### Training
```bash
accelerate launch train_mn_proposed.py
```

#### Inference
The `inference.py` script allows you to convert individual audio files with simple command line arguments.


1. **Timbre Conversion** - Convert timbre to match reference audio:
```bash
python inference.py \
    --orig_audio audios/sample_orig.wav \
    --ref_audio audios/sample_ref.wav \
    --convert_type timbre \
    --ckpt_path /path/to/your/checkpoints \
    --output_path output_timbre.wav
```

2. **ADSR Conversion** - Convert ADSR envelope to match reference:
```bash
python inference.py \
    --orig_audio audios/sample_orig.wav \
    --ref_audio audios/sample_ref.wav \
    --convert_type adsr \
    --ckpt_path /path/to/your/checkpoints \
    --output_path output_adsr.wav
```

3. **Both Timbre and ADSR** - Convert both timbre and ADSR envelope characteristics:
```bash
python inference.py \
    --orig_audio audios/sample_orig.wav \
    --ref_audio audios/sample_ref.wav \
    --convert_type both \
    --ckpt_path /path/to/your/checkpoints \
    --output_path output_both.wav
```


**Command Line Arguments:**
- `--orig_audio` (required): Path to original audio file
- `--ref_audio` (optional): Path to reference audio file (required for timbre/adsr/both conversions)
- `--output_path` (optional): Output file path (default: `converted_{convert_type}.wav`)
- `--convert_type` (optional): Conversion type - `timbre`, `adsr`, `both`, or `reconstruction` (default: `timbre`)
- `--iter` (optional): Checkpoint iteration to load (default: 400000 for latest)
- `--ckpt_path` (optional): Path to checkpoint directory (overrides config file)



## Evaluation Results

| Method | MSTFT ↓ | LRMSD ↓ | F0RMSE ↓ | TMOS ↑ | ADSRMOS ↑ | CMOS ↑ |
|--------|---------|---------|----------|--------|-----------|--------|
| Ground Truth | -- | -- | -- | 4.08 | 3.96 | 4.25 |
| SS-VAE | 7.22 | 0.92 | 641.62 | 2.20 | 2.25 | 3.41 |
| CTD | 5.69 | 0.89 | 583.01 | 2.34 | 2.48 | 1.86 |
| **SynthCloner (ours)** | **3.00** | **0.17** | **20.64** | **3.91** | **3.94** | **4.11** |
| -- w/o ADSR envelope path | 3.84 | 0.42 | 29.04 | 3.20 | 2.43 | 3.87 |

**Objective Metrics:**
- **MSTFT**: Multi-Scale STFT Loss (lower is better)
- **LRMSD**: Log RMS Distance (lower is better)  
- **F0RMSE**: F0 Root Mean Square Error (lower is better)

**Subjective Metrics:**
- **TMOS**: Timbre Mean Opinion Score (higher is better)
- **ADSRMOS**: ADSR Mean Opinion Score (higher is better)
- **CMOS**: Comparative Mean Opinion Score (higher is better)


## Citation

If you use this work in your research, please cite:

```bibtex
@article{synthcloner,
  title={SynthCloner: Synthesizer Preset Conversion via Factorized Codec with Disentangled Timbre and ADSR Control},
  author={Liu, Jeng-Yue and Hsu, Ting-Chao and Yeh, Yen-Tung and Su, Li and Yang, Yi-Hsuan},
  journal=TBD,
  year={2025}
}
```


<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Contact

- **Jeng-Yue Liu**: [philip910323@gmail.com](mailto:philip910323@gmail.com)
- **Project Page**: [https://buffett0323.github.io/synthcloner/](https://buffett0323.github.io/synthcloner/)
---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/buffett0323/synthcloner?style=social)](https://github.com/buffett0323/synthcloner)
[![GitHub forks](https://img.shields.io/github/forks/buffett0323/synthcloner?style=social)](https://github.com/buffett0323/synthcloner)

</div>
