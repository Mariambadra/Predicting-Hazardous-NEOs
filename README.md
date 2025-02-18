# Near-Earth Objects Hazard Prediction

## Project Overview
This project aims to predict whether Near-Earth Objects (NEOs) pose a potential hazard using NASA's dataset. The implementation includes data cleaning, exploratory analysis, preprocessing, and machine learning modeling using Random Forest Classifier.

## Dataset
- Source: NASA NEO dataset (1910-2024)
- Original Features: 
  `neo_id`, `name`, `absolute_magnitude`, `estimated_diameter_min/max`, 
  `relative_velocity`, `miss_distance`, `is_hazardous`
- Cleaned Dataset: 338,171 entries

## Key Workflow Steps

### 1. Data Cleaning
- Removed 28 rows with missing values
- Dropped redundant columns:
  - `orbiting_body` (single unique value)
  - `neo_id` & `name` (high cardinality)
  - `estimated_diameter_min/max` (collinear with absolute_magnitude)

### 2. Feature Engineering
- Converted target variable: `is_hazardous` (bool → int)
- Addressed class imbalance using SMOTE
- Standardized features using StandardScaler

### 3. Model Development
```python
# Final Model Architecture
RandomForestClassifier(
    max_depth=10,
    min_samples_split=20,
    class_weight='balanced',
    random_state=42
)
