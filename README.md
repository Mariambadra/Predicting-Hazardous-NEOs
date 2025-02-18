# 🌍 Near-Earth Objects Hazard Prediction System

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

![Project Banner](https://defence-industry-space.ec.europa.eu/sites/default/files/styles/oe_theme_full_width_banner_4_1/public/2023-08/NEO_HEADER%201602x530.png.webp?itok=rbmsPd05)  
*Classifying potentially hazardous asteroids using machine learning*
![Workflow Diagram](https://via.placeholder.com/800x400?text=Data+Processing+and+Model+Workflow)

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Key Features](#-key-features)
- [Technical Implementation](#-technical-implementation)
- [Results & Metrics](#-results--metrics)
- [Deployment](#-deployment)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Limitations & Future Work](#-limitations--future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Project Overview

**Objective**: Develop a binary classification system to predict asteroid hazard potential using NASA's NEO dataset  
**Significance**: Supports planetary defense initiatives by automating risk assessment of near-Earth objects  
**Key Challenges**:
- Severe class imbalance (17:1 non-hazardous vs hazardous)
- High-dimensional feature space
- Physically meaningful feature engineering

---

## 📊 Dataset Description

### Source Data
- **Original Dataset**: [NASA CNEOS Close Approach Data](https://cneos.jpl.nasa.gov/ca/)
- **Temporal Coverage**: 1910-2024
- **Initial Size**: 338,171 observations
- **License**: NASA Open Data Agreement

### Final Feature Set
| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `absolute_magnitude` | Object's intrinsic brightness | mag | 10.3-32.6 |
| `relative_velocity` | Approach speed | km/h | 1,380-152,700 |
| `miss_distance` | Closest approach distance | km | 5,000-750M |
| `is_hazardous` | Hazard classification | bool | 0/1 |

---

## 🔑 Key Features

1. **Advanced Imbalance Handling**
   - SMOTE oversampling (minority class)
   - Class-weighted Random Forest
   - Stratified k-fold validation

2. **Feature Engineering**
   - Removed collinear features (r=1.0)
   - Standardized physical parameters
   - Astrophysical domain validation

3. **Production-Ready Pipeline**
   - Model serialization with joblib
   - Input validation constraints
   - Comprehensive error handling

---

## ⚙️ Technical Implementation

### Preprocessing Pipeline
```python
# Class Balancing
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_res)
```

### Optimized Model Architecture
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    max_depth=10,
    min_samples_split=20,
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
```

### Hyperparameter Search Space
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}
```

---

## 📊 Results & Metrics

### Performance Summary

| Metric  | Value  | Description  |
|---------|--------|-------------|
| AUC-ROC | 0.9551 | Area Under ROC Curve |
| F1 Score | 0.90 | Harmonic Mean of Precision/Recall |
| Recall | 0.96 | Hazardous Class Detection Rate |
| Precision | 0.84 | Hazardous Prediction Accuracy |

### Confusion Matrix
```
               Predicted Safe  Predicted Hazardous
Actual Safe         48,237             10,803
Actual Hazard       2,438              56,526
```
---

## 🚀 Deployment
- Model is serialized using `joblib` for fast loading
