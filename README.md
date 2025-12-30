# Kvasir Medical Image Segmentation

A comprehensive medical image segmentation project implementing **UNet**, **TransUNet**, and **Ensemble Fusion** models for polyp detection and segmentation using the Kvasir-SEG dataset.

## 🎯 Project Overview

This project focuses on automated polyp segmentation in colonoscopy images, which is crucial for early detection and diagnosis of colorectal cancer. We implemented and compared three state-of-the-art deep learning architectures:

1. **UNet** - Classic encoder-decoder architecture with skip connections
2. **TransUNet** - Transformer-based encoder for better feature extraction
3. **Ensemble Fusion** - Combined predictions from UNet and TransUNet for improved accuracy

## 📊 Performance Results

### Model Comparison - Complete Metrics

#### Validation Set Results

| Model | Accuracy | IoU (Jaccard) | Dice (F1) | Recall | Precision |
|-------|----------|---------------|-----------|--------|-----------|
| **UNet** | 97.10% | 82.47% | 88.87% | 90.24% | 90.90% |
| **TransUNet** | 97.08% | 82.10% | 88.34% | 88.90% | 91.34% |

#### Test Set Results (Kvasir-SEG)

| Model | Accuracy | IoU (Jaccard) | Dice (F1) | Recall | Precision |
|-------|----------|---------------|-----------|--------|-----------|
| **UNet** | 96.65% | 83.53% | 89.33% | 91.87% | 90.89% |
| **TransUNet** | 96.56% | 83.92% | 89.58% | 91.73% | 91.65% |
| **Ensemble Fusion** | **96.87%** | **84.78%** | **90.18%** | **91.95%** | **92.37%** |

#### Sessile Polyp Dataset Results (Kvasir-Sessile)

| Model | Accuracy | IoU (Jaccard) | Dice (F1) | Recall | Precision |
|-------|----------|---------------|-----------|--------|-----------|
| **UNet** | 98.11% | 80.09% | 87.42% | 89.27% | 89.10% |
| **TransUNet** | 98.06% | 79.35% | 86.83% | 89.10% | 88.90% |
| **Ensemble Fusion** | **98.18%** | **80.38%** | **87.74%** | **89.44%** | **89.65%** |

### Key Insights

- **Best Model**: Ensemble Fusion consistently achieves the highest performance across all datasets
- **Test Set**: 84.78% IoU and 90.18% Dice coefficient on Kvasir-SEG
- **Generalization**: Excellent performance (98.18% accuracy) on sessile polyps demonstrates robust feature learning
- **Ensemble Advantage**: Simple averaging of predictions improves results by leveraging strengths of both architectures

## 🏗️ Architecture Details

### UNet
- **Encoder**: MiT-B0 (Mix Transformer) with ImageNet pre-training
- **Framework**: segmentation-models-pytorch (smp)
- **Loss Function**: Dice Loss
- **Optimizer**: Adam
- **Image Size**: 256x256

### TransUNet
- **Encoder**: Transformer-based encoder
- **Framework**: segmentation-models-pytorch (smp.Unet)
- **Loss Function**: Dice Loss
- **Optimizer**: Adam
- **Image Size**: 256x256

### Ensemble Fusion
- **Method**: Simple averaging of predictions
- **Formula**: `(output_unet + output_transunet) / 2`
- **Threshold**: 0.5 for binary segmentation

## 📁 Project Structure

```
kvasir-medical-segmentation/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── notebooks/
│   └── Kvasir.ipynb                  # Main training and evaluation notebook
├── models/
│   ├── unet_fixed_split_best_weights.pth      # Trained UNet weights
│   └── transunet_fixed_split_best_weights.pth # Trained TransUNet weights
└── results/
    └── performance_metrics.md         # Detailed performance analysis
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Google Colab account (if running in cloud)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yusufsakirr1/kvasir-medical-segmentation.git
cd kvasir-medical-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Kvasir-SEG dataset:
   - Visit: https://www.kaggle.com/datasets/dankok/kvasir-seg
   - Extract to your preferred location
   - Update dataset paths in the notebook

## 🚀 Usage

### Training

Open `notebooks/Kvasir.ipynb` in Jupyter or Google Colab and follow these steps:

1. **Setup Environment**: Install required libraries
2. **Mount Google Drive**: (If using Colab) to access datasets
3. **Configure Paths**: Update dataset base path
4. **Train UNet**: Run UNet training cells
5. **Train TransUNet**: Run TransUNet training cells
6. **Evaluate Ensemble**: Run ensemble evaluation cells

### Inference

To use pre-trained models for inference:

```python
import torch
import segmentation_models_pytorch as smp

# Load UNet model
model_unet = smp.Unet(
    encoder_name='mit_b0',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)
model_unet.load_state_dict(torch.load('models/unet_fixed_split_best_weights.pth'))
model_unet.eval()

# Load TransUNet model
model_transunet = smp.Unet(
    encoder_name='transformer-based-encoder',  # Adjust based on your config
    in_channels=3,
    classes=1
)
model_transunet.load_state_dict(torch.load('models/transunet_fixed_split_best_weights.pth'))
model_transunet.eval()

# Ensemble prediction
with torch.no_grad():
    output_unet = model_unet(image)
    output_transunet = model_transunet(image)
    ensemble_output = (output_unet + output_transunet) / 2
    prediction = (ensemble_output > 0.5).float()
```

## 📊 Dataset

### Kvasir-SEG
- **Description**: Gastrointestinal polyp segmentation dataset
- **Images**: 1000 polyp images with corresponding masks
- **Resolution**: Various (resized to 256x256 for training)
- **Split**: 80% training, 20% validation
- **Source**: https://www.kaggle.com/datasets/dankok/kvasir-seg

### Kvasir-Sessile
- **Description**: Subset focusing on sessile polyps
- **Usage**: Additional testing and validation
- **Purpose**: Evaluate model generalization

## 🧪 Training Configuration

- **Image Size**: 256x256 pixels
- **Batch Size**: Configurable (typically 8-16)
- **Learning Rate**: 0.0001 (Adam optimizer)
- **Loss Function**: Dice Loss
- **Metrics**: IoU (Jaccard Index), F1-Score (Dice Coefficient), Accuracy
- **Data Augmentation**: Albumentations library
  - Horizontal/Vertical Flip
  - Rotation
  - Random brightness/contrast
  - Elastic transform

## 📈 Experiment Tracking

This project uses **Weights & Biases (wandb)** for experiment tracking:
- Training/validation loss curves
- Metric evolution over epochs
- Model performance comparison
- Hyperparameter logging

## 🔬 Key Features

- **Multiple Architectures**: Comparison of UNet, TransUNet, and Ensemble approaches
- **Transfer Learning**: ImageNet pre-trained encoders for better initialization
- **Data Augmentation**: Robust augmentation pipeline using Albumentations
- **Ensemble Learning**: Improved performance through model combination
- **Comprehensive Metrics**: IoU, F1-Score, and Accuracy evaluation
- **Visualization**: Prediction visualization with ground truth comparison

## 👤 Author

**Yusuf Sakirri**
- GitHub: [@yusufsakirr1](https://github.com/yusufsakirr1)

## 🙏 Acknowledgments

- **Kvasir-SEG Dataset**: Simula Research Laboratory
- **segmentation-models-pytorch**: Pavel Yakubovskiy
- **PyTorch Team**: For the excellent deep learning framework
- **Google Colab**: For providing free GPU resources

## 📧 Contact

For questions or collaborations, please open an issue or reach out via GitHub.

---

**Note**: Model weight files (`.pth`) are not included in this repository due to file size constraints. They can be downloaded from [releases](https://github.com/yusufsakirr1/kvasir-medical-segmentation/releases) or regenerated by running the training notebook.
