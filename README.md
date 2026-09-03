# PaDuM: Patch-Based Dual-Stream Network with CNN and Mamba for Time Series Forecasting

<div align="center">

[![ICASSP 2026](https://img.shields.io/badge/Conference-ICASSP%202026-blue)](https://2026.ieeeicassp.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-green)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Junliang Tao, Li Cao, Hongbing Wang, Chenhao Xie, Jian Li, Liang Zhou**

*School of Big Data and Computer Science, Guizhou Normal University, Guiyang, China*

</div>

## 📝 Abstract

Recently, Transformer-based models have shown limitations in efficiency, while Mamba offers strong potential with its linear complexity and selective mechanism. Meanwhile, convolutional neural networks (CNNs) excel at capturing local dependencies. To integrate their complementary strengths, we propose **PaDuM**, a **Pa**tch-Based **Du**al-Stream Network with CNN and **M**amba. PaDuM employs exponential moving average (EMA) to decouple sequences into trend and seasonal components, which are then processed through patching and modeled jointly by CNN and Mamba. Furthermore, we design a Sigmoid-based weight decay loss to emphasize recent predictions and enhance stability. Extensive experiments on eight real-world datasets from electricity, traffic, and meteorology domains demonstrate that PaDuM achieves state-of-the-art performance with strong robustness and generalization.

## 🌟 Highlights

- **First Patch-Based Dual-Stream Architecture**: PaDuM is the first model combining CNN and Mamba in a patch-based dual-stream framework for time series forecasting.

- **EMA Decomposition**: Uses Exponential Moving Average to decouple time series into trend and seasonal components, providing more discriminative representations.

- **Sigmoid Loss Function**: Novel Sigmoid-based loss that emphasizes short-term predictions, improving MAE while maintaining stable optimization.

- **State-of-the-Art Performance**: Achieves best MAE on most of 8 real-world datasets across electricity, traffic, and meteorology domains.

- **Efficient Architecture**: Linear complexity with Mamba and lightweight CNN, achieving favorable accuracy-efficiency trade-off.

## 🏗️ Architecture

<div align="center">
<img src="./figures/fig2.png" alt="PaDuM Framework" width="70%"/>
<p><em>Overview of the PaDuM framework. The input time series is decomposed into seasonal and trend components, modeled by CNN and Mamba streams, respectively.</em></p>
</div>

### Key Components:

1. **EMA Decomposition**: Separates input into trend (long-term dependencies) and seasonal (periodic patterns) components

2. **CNN Stream**: Processes seasonal component using depthwise separable convolutions to capture local patterns

3. **Mamba Stream**: Models trend component with state space models for efficient long-range dependency modeling

4. **Fusion Layer**: Concatenates and merges features from both streams via fully connected layer

## 📊 Main Results

### Long-term Forecasting (Look-back Window L=96)

| Models | PaDuM (Ours) | xPatch | S-Mamba | CARD | TimeMixer | DLinear | PatchTST | FEDformer | Autoformer |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1st Count** | **9** | 2 | 2 | 2 | 3 | 0 | 1 | 1 | 0 |

<details>
<summary>📈 Detailed Results (Click to expand)</summary>

| Dataset | Metric | PaDuM | xPatch | S-Mamba | CARD | TimeMixer | DLinear | PatchTST | FEDformer | Autoformer |
|---------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ETTh1 | MSE | 0.448 | 0.444 | 0.457 | 0.449 | 0.458 | 0.460 | 0.445 | 0.436 | 0.493 |
| ETTh1 | MAE | **0.430** | **0.430** | 0.452 | 0.435 | 0.444 | 0.453 | 0.436 | 0.456 | 0.483 |
| ETTh2 | MSE | 0.364 | 0.369 | 0.383 | **0.360** | 0.385 | 0.496 | 0.362 | 0.424 | 0.442 |
| ETTh2 | MAE | **0.387** | 0.392 | 0.408 | 0.389 | 0.405 | 0.479 | 0.393 | 0.443 | 0.457 |
| ETTm1 | MSE | 0.387 | 0.390 | 0.398 | 0.391 | 0.386 | 0.406 | **0.381** | 0.408 | 0.533 |
| ETTm1 | MAE | **0.386** | **0.386** | 0.406 | 0.390 | 0.400 | 0.410 | 0.395 | 0.431 | 0.492 |
| ETTm2 | MSE | **0.278** | 0.280 | 0.290 | **0.278** | **0.278** | 0.311 | 0.280 | 0.292 | 0.323 |
| ETTm2 | MAE | **0.319** | 0.321 | 0.333 | 0.321 | 0.325 | 0.368 | 0.326 | 0.345 | 0.367 |
| Electricity | MSE | 0.178 | 0.198 | **0.172** | 0.177 | 0.182 | 0.209 | 0.188 | 0.188 | 0.234 |
| Electricity | MAE | **0.263** | 0.276 | 0.268 | 0.268 | 0.272 | 0.296 | 0.275 | 0.275 | 0.341 |
| Solar | MSE | 0.250 | 0.294 | 0.244 | 0.243 | **0.234** | 0.330 | 0.250 | 0.292 | 0.878 |
| Solar | MAE | **0.243** | 0.273 | 0.275 | 0.241 | 0.292 | 0.402 | 0.283 | 0.381 | 0.705 |
| Traffic | MSE | 0.524 | 0.494 | **0.412** | 0.440 | 0.504 | 0.625 | 0.462 | 0.610 | 0.614 |
| Traffic | MAE | **0.270** | 0.293 | 0.278 | 0.273 | 0.301 | 0.384 | 0.290 | 0.380 | 0.378 |
| Weather | MSE | 0.250 | 0.254 | 0.252 | 0.248 | **0.245** | 0.267 | 0.258 | 0.309 | 0.342 |
| Weather | MAE | **0.268** | 0.272 | 0.277 | 0.270 | 0.276 | 0.317 | 0.280 | 0.356 | 0.384 |

</details>

### Performance Highlights

<div align="center">
<img src="./figures/fig1.png" alt="Performance Comparison" width="60%"/>
<p><em>Average MAE comparison between PaDuM and state-of-the-art baselines with look-back window size T=96.</em></p>
</div>

## 🔬 Ablation Studies

### Architecture Design

<div align="center">
<img src="./figures/fig3.png" alt="Architecture Ablation" width="50%"/>
</div>

| Variant | Description |
|---------|-------------|
| **PaDuM** | Seasonality → CNN, Trend → Mamba |
| Reversed | Seasonality → Mamba, Trend → CNN |
| CNN Only | Both → CNN |
| Mamba Only | Both → Mamba |

**Finding**: Original PaDuM configuration achieves best performance by leveraging CNN's strength in local patterns and Mamba's efficiency in long-term dependencies.

### Sigmoid Loss Effectiveness

<div align="center">
<img src="./figures/fig4.png" alt="Sigmoid Loss Ablation" width="45%"/>
</div>

The Sigmoid Loss enhances forecasting performance with notable improvements in long-term accuracy, confirming its effectiveness in focusing on recent predictions.

## ⚡ Efficiency Analysis

<div align="center">
<img src="./figures/fig5.png" alt="Training Efficiency" width="45%"/>
<img src="./figures/fig6.png" alt="Inference Efficiency" width="45%"/>
</div>

- **Training**: PaDuM achieves lowest MAE with fewer parameters and shorter training time
- **Inference**: One-pass latency remains nearly constant (~15ms), GPU memory increases linearly with input length

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.1.0+
- CUDA 11.8+ (for Mamba support)

### Setup

```bash
# Clone the repository
git clone https://github.com/T-DXVN/PaDuM.git
cd PaDuM

# Create conda environment (recommended)
conda create -n padum python=3.8
conda activate padum

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install Mamba
pip install mamba-ssm==1.2.0

# Install other dependencies
pip install -r requirements.txt
```

### Dependencies

- `torch==2.1.0`
- `mamba-ssm==1.2.0`
- `numpy==1.26.4`
- `scikit-learn==1.3.0`
- `matplotlib==3.7.0`
- `reformer-pytorch==1.4.4`

## 🚀 Quick Start

### Data Preparation

Download the datasets from [Google Drive](https://drive.google.com/drive/folders/13CgB3IDNNpFGLem5MWdBXPcDh72D1sSm) or [Baidu Drive](https://pan.baidu.com/s/1r318GGN-0tEbGh0mwjNHow?pwd=ic84).

Place the data in `./dataset/` folder.

### Training

```bash
# Train on ETTh1 dataset with prediction length 96
python run.py \
    --is_training 1 \
    --model_id ETTh1_96_ema \
    --model PaDuM \
    --data ETTh1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --patch_len 16 \
    --stride 8 \
    --enc_in 7 \
    --d_model 256 \
    --d_state 2 \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --ma_type ema \
    --alpha 0.3 \
    --beta 0.3

# Or use the provided script to train on all datasets
bash scripts/all.sh
```

### Evaluation

```bash
# Evaluate trained model
python run.py \
    --is_training 0 \
    --model_id ETTh1_96_ema \
    --model PaDuM \
    --data ETTh1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --patch_len 16 \
    --stride 8 \
    --enc_in 7 \
    --d_model 256 \
    --d_state 2 \
    --ma_type ema
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name | `PaDuM` |
| `--data` | Dataset name | `ETTh1` |
| `--data_path` | Data file name | `ETTh1.csv` |
| `--seq_len` | Look-back window | `96` |
| `--label_len` | Label length | `48` |
| `--pred_len` | Prediction horizon | `96` |
| `--features` | Feature type (`M`/`S`/`MS`) | `M` |
| `--enc_in` | Number of input features | `7` |
| `--d_model` | Model dimension | `256` |
| `--d_state` | Mamba state dimension | `2` |
| `--patch_len` | Patch length | `16` |
| `--stride` | Patch stride | `8` |
| `--ma_type` | Moving average type (`ema`/`dema`/`reg`) | `ema` |
| `--alpha` | EMA smoothing factor | `0.3` |
| `--beta` | DEMA smoothing factor | `0.3` |
| `--learning_rate` | Learning rate | `0.0001` |
| `--batch_size` | Batch size | `32` |
| `--train_epochs` | Training epochs | `100` |
| `--Slope` | Sigmoid loss slope | `0.5` |
| `--Center` | Sigmoid loss center | `10.0` |
| `--lower_bound` | Sigmoid loss lower bound | `0.2` |
| `--revin` | Use Reversible Instance Norm (`1`=True, `0`=False) | `1` |

## 📁 Project Structure

```
PaDuM/
├── models/
│   └── PaDuM.py           # Main PaDuM model
├── layers/
│   ├── net_CNN.py         # CNN stream implementation
│   ├── net_Mamba.py       # Mamba stream implementation
│   ├── network.py         # Network utilities
│   ├── ema.py             # Exponential Moving Average
│   ├── dema.py            # Double EMA
│   ├── decomp.py          # Decomposition layer
│   └── revin.py           # Reversible Instance Normalization
├── data_provider/
│   ├── data_factory.py    # Data loading factory
│   └── data_loader.py     # Dataset implementations
├── exp/
│   ├── exp_basic.py       # Base experiment class
│   └── exp_main.py        # Main experiment logic
├── utils/
│   ├── tools.py           # Utility functions
│   ├── metrics.py         # Evaluation metrics (MSE, MAE)
│   └── timefeatures.py    # Time feature engineering
├── scripts/
│   ├── all.sh             # Full training script
│   └── sigmoid_ablation.sh # Sigmoid loss ablation
├── ablation/              # Ablation study implementations
├── figures/               # Paper figures
├── dataset/               # Dataset folder (not included)
├── run.py                 # Main entry point
├── generate_table.py      # Results table generator
├── requirements.txt       # Dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{tao2026padum,
  title={PaDuM: Patch-Based Dual-Stream Network with CNN and Mamba for Time Series Forecasting},
  author={Tao, Junliang and Cao, Li and Wang, Hongbing and Xie, Chenhao and Li, Jian and Zhou, Liang},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## 🙏 Acknowledgements

This work was supported by:
- National Natural Science Foundation of China (U22A2026, 62072097)
- QIANHEHE PLATFORM TALENT BQW[2024]015
- GZNU[2024]01

We thank the developers of [Time Series Library (TS-Lib)](https://github.com/thuml/Time-Series-Library) for their excellent codebase.

## 📧 Contact

For questions or collaborations, please contact:

- **Corresponding Author**: Hongbing Wang (hbwang@gznu.edu.cn)
- **First Author**: Junliang Tao

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
