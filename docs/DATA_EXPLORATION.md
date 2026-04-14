# Data Exploration Report

## Dataset Overview
- **Source:** TrashNet Dataset
- **Total Images:** [Fill after running notebook]
- **Categories:** 6 (Cardboard, Glass, Metal, Paper, Plastic, Trash)

## Key Findings

### 1. Class Distribution
- **Most Common:** [Fill from notebook]
- **Least Common:** [Fill from notebook]
- **Imbalance Ratio:** [Fill from notebook]
- **Action Needed:** Data augmentation for minority classes

### 2. Image Properties
- **Average Dimensions:** ~512 x 384 pixels
- **Aspect Ratio:** Varies (need standardization)
- **Format:** JPG
- **Quality:** Generally good, some low-light images

### 3. Challenges Identified
- [ ] Class imbalance (some categories have fewer images)
- [ ] Varying image sizes (need resizing)
- [ ] Different lighting conditions
- [ ] Some images have multiple objects

### 4. Next Steps
- Implement data augmentation
- Standardize image size to 224x224 (for EfficientNet)
- Split data: 70% train, 15% val, 15% test