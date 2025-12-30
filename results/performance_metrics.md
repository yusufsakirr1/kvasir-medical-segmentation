# Performance Metrics

## Model Performance Summary

### Complete Performance Results

#### UNet Model

**Validation Set Performance:**
- **Accuracy**: 97.10%
- **IoU (Jaccard Index)**: 82.47%
- **F1-Score (Dice Coefficient)**: 88.87%
- **Recall**: 90.24%
- **Precision**: 90.90%

**Test Set Performance:**
- **Accuracy**: 96.65%
- **IoU (Jaccard Index)**: 83.53%
- **F1-Score (Dice Coefficient)**: 89.33%
- **Recall**: 91.87%
- **Precision**: 90.89%

**Sessile Polyp Dataset Performance:**
- **Accuracy**: 98.11%
- **IoU (Jaccard Index)**: 80.09%
- **F1-Score (Dice Coefficient)**: 87.42%
- **Recall**: 89.27%
- **Precision**: 89.10%

**Model Details:**
- **Architecture**: UNet with MiT-B0 encoder
- **Encoder Weights**: ImageNet pre-trained
- **Training**: Fixed split with data augmentation
- **Model File**: `unet_fixed_split_best_weights.pth`

#### TransUNet Model

**Validation Set Performance:**
- **Accuracy**: 97.08%
- **IoU (Jaccard Index)**: 82.10%
- **F1-Score (Dice Coefficient)**: 88.34%
- **Recall**: 88.90%
- **Precision**: 91.34%

**Test Set Performance:**
- **Accuracy**: 96.56%
- **IoU (Jaccard Index)**: 83.92%
- **F1-Score (Dice Coefficient)**: 89.58%
- **Recall**: 91.73%
- **Precision**: 91.65%

**Sessile Polyp Dataset Performance:**
- **Accuracy**: 98.06%
- **IoU (Jaccard Index)**: 79.35%
- **F1-Score (Dice Coefficient)**: 86.83%
- **Recall**: 89.10%
- **Precision**: 88.90%

**Model Details:**
- **Architecture**: UNet with Transformer-based encoder
- **Training**: Fixed split with data augmentation
- **Model File**: `transunet_fixed_split_best_weights.pth`

#### Ensemble Fusion Model

**Test Set Performance:**
- **Accuracy**: 96.87%
- **IoU (Jaccard Index)**: **84.78%**
- **F1-Score (Dice Coefficient)**: **90.18%**
- **Recall**: 91.95%
- **Precision**: 92.37%

**Sessile Polyp Dataset Performance:**
- **Accuracy**: 98.18%
- **IoU (Jaccard Index)**: 80.38%
- **F1-Score (Dice Coefficient)**: 87.74%
- **Recall**: 89.44%
- **Precision**: 89.65%

**Model Details:**
- **Method**: Simple averaging of UNet and TransUNet outputs
- **Formula**: `(output_unet + output_transunet) / 2`
- **Threshold**: 0.5 for binary segmentation

## Key Findings

1. **Best Performance**: The Ensemble Fusion model achieved the highest scores across all metrics
   - **Test Set**: 84.78% IoU, 90.18% Dice, 92.37% Precision
   - **Sessile Set**: 98.18% Accuracy, 80.38% IoU, 87.74% Dice
   - Ensemble consistently outperforms individual models

2. **Model Comparison**:
   - **Test Set**: TransUNet (83.92% IoU) slightly outperformed UNet (83.53% IoU)
   - **Validation Set**: UNet (82.47% IoU) performed better than TransUNet (82.10% IoU)
   - Combining both models through ensemble learning provided the best results
   - The ensemble approach leverages the strengths of both architectures

3. **Generalization**:
   - All models demonstrated excellent generalization on the Kvasir-Sessile dataset
   - High accuracy (>98%) on sessile polyps indicates robust feature learning
   - Ensemble model achieved the best sessile polyp detection accuracy (98.18%)

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
