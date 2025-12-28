# Performance Metrics

## Model Performance Summary

### Kvasir-SEG Test Set Results

#### UNet Model
- **Architecture**: UNet with MiT-B0 encoder
- **Encoder Weights**: ImageNet pre-trained
- **IoU (Jaccard Index)**: 81.38%
- **F1-Score (Dice Coefficient)**: 89.73%
- **Training**: Fixed split with data augmentation
- **Model File**: `unet_fixed_split_best_weights.pth`

#### TransUNet Model
- **Architecture**: UNet with Transformer-based encoder
- **IoU (Jaccard Index)**: 79.03%
- **F1-Score (Dice Coefficient)**: 88.09%
- **Training**: Fixed split with data augmentation
- **Model File**: `transunet_fixed_split_best_weights.pth`

#### Ensemble Fusion Model
- **Method**: Simple averaging of UNet and TransUNet outputs
- **Formula**: `(output_unet + output_transunet) / 2`
- **IoU (Jaccard Index)**: **84.78%**
- **F1-Score (Dice Coefficient)**: **90.18%**
- **Threshold**: 0.5 for binary segmentation

## Key Findings

1. **Best Performance**: The Ensemble Fusion model achieved the highest scores on both metrics
   - 3.4% improvement in IoU over standalone UNet
   - 5.75% improvement in IoU over standalone TransUNet
   - 0.45% improvement in F1-Score over standalone UNet

2. **Model Comparison**:
   - UNet outperformed TransUNet in this specific task
   - Combining both models through ensemble learning provided the best results
   - The ensemble approach leverages the strengths of both architectures

3. **Generalization**: Additional testing on Kvasir-Sessile dataset demonstrates model robustness on sessile polyp detection

## Training Configuration

- **Dataset**: Kvasir-SEG (1000 images)
- **Train/Val Split**: 80/20
- **Image Size**: 256x256
- **Loss Function**: Dice Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.0001
- **Data Augmentation**:
  - Horizontal/Vertical Flip
  - Rotation
  - Random Brightness/Contrast
  - Elastic Transform

## Evaluation Metrics

### IoU (Intersection over Union / Jaccard Index)
- Measures overlap between predicted and ground truth masks
- Formula: `IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)`
- Range: 0-1 (0-100%)
- Higher is better

### F1-Score (Dice Coefficient)
- Harmonic mean of precision and recall
- Formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- Equivalent to: `Dice = 2 * |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)`
- Range: 0-1 (0-100%)
- Higher is better

## Visualization

Sample predictions and visualizations can be found in the main notebook: `notebooks/Kvasir.ipynb`

---

**Last Updated**: December 2025
