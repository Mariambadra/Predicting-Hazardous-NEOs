# Near-Earth Objects (NEO) Hazard Prediction System

![Project Banner](https://defence-industry-space.ec.europa.eu/sites/default/files/styles/oe_theme_full_width_banner_4_1/public/2023-08/NEO_HEADER%201602x530.png.webp?itok=rbmsPd05)  
*Classifying potentially hazardous asteroids using machine learning*

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Methodology](#methodology)
4. [Technical Implementation](#technical-implementation)
5. [Results & Interpretation](#results--interpretation)
6. [Deployment](#deployment)
7. [Repository Structure](#repository-structure)
8. [Getting Started](#getting-started)
9. [Limitations & Future Work](#limitations--future-work)
10. [Contributing](#contributing)

---

## Project Overview
**Objective**: Develop a binary classification system to predict asteroid hazard potential using orbital characteristics and physical properties.  
**Significance**: Supports NASA's planetary defense initiatives by automating risk assessment of NEOs.  
**Key Features**:
- Handles severe class imbalance (17:1 hazardous:non-hazardous ratio)
- Implements robust data leakage prevention
- Achieves 95.5% AUC-ROC score
- Deployable model for real-time predictions

---

## Dataset Description
### Source
- **Original Dataset**: [NASA NEO Earth Close Approaches](https://cneos.jpl.nasa.gov/ca/)
- **Time Range**: 1910-2024
- **Size**: 338,171 observations
- **Original Features**:
  ```csv
  neo_id, name, absolute_magnitude, estimated_diameter_min, 
  estimated_diameter_max, relative_velocity, miss_distance, 
  orbiting_body, is_hazardous
