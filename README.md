# 🚖 NYC Taxi Trip Duration Prediction
### Advanced Mobility Analytics with XGBoost & K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

## 📖 Overview

In the logistics and mobility sector, accurate time estimation is the cornerstone of fleet efficiency and user satisfaction. This project analyzes over **1.4 million taxi trips** in New York City to predict the total trip duration.

Unlike standard regression approaches, this project utilizes **Spatial Clustering (K-Means)** to define functional traffic zones and **Cyclical Temporal Encoding** to capture the continuous nature of time, resulting in a highly competitive RMSLE score.

---

## 🏆 Key Results

The final model, utilizing **Log-Target Transformation** and **Hyperparameter Tuning** via RandomizedSearchCV, achieved a competitive score, breaking the **0.40 RMSLE barrier**.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **RMSLE (Key Metric)** | **0.39892** | **Competitive Tier.** Errors on long/short trips are balanced effectively. |
| **MAE** | **208.28 sec** | The model is off by only **~3.5 minutes** on average. |
| **RMSE** | 327.78 sec | Indicates robustness against extreme outliers. |
| **R² Score** | 0.7374 | The model explains ~74% of the variance in trip duration. |

---

## 🧠 Methodology & Feature Engineering

### 1. Data Cleaning & Outlier Detection
Raw GPS data is noisy. Rigorous filtering was applied to ensure physical viability:
* **Geobounding:** Trips restricted to NYC coordinates (Lat: 40.63–40.85, Lon: -74.03–-73.77).
* **Speed/Duration:** Removed impossible trips (Speed > 90km/h, Duration > 120 min).
* **Sanity Checks:** Removed trips with 0 passengers or 0 distance.

### 2. Spatial Engineering: Functional Zones (K-Means)
Raw Latitude/Longitude coordinates are inefficient for tree-based models to split.
* **Strategy:** Performed **K-Means Clustering (K=15)** on pickup locations to create discrete "Traffic Zones."
* **Result:** Instead of raw coordinates, the model sees "Trip from Cluster 3 to Cluster 9," allowing it to learn route-specific congestion patterns (e.g., JFK Airport to Midtown).
* **Visualization:** Developed dynamic flow maps showing aggregate trip volume between zone centroids.

### 3. Temporal Engineering: Cyclical Features
Time is a circle, not a straight line (23:00 is mathematically close to 00:00).
* **Strategy:** Transformed `Hour` and `Month` into 2D coordinates using Sine and Cosine functions.
* **Impact:** Removed the "midnight cliff" problem, allowing the model to understand the continuity of late-night traffic.

### 4. Target Transformation
Taxi trip duration follows a **Right-Skewed (Log-Normal)** distribution.
* **Strategy:** Trained the model on `np.log1p(trip_duration)` rather than raw seconds.
* **Result:** Significantly reduced the impact of massive outliers, dropping RMSLE from **0.42** to **0.39**.

---

## ⚙️ Model Pipeline (XGBoost)

The final solution uses a robust **Scikit-Learn Pipeline** to ensure no data leakage ("Fit on Train, Transform on Test").

* **Preprocessing:**
    * *Categorical:* One-Hot Encoding for `VendorID`, `StoreFlag`, and **Clusters**.
    * *Numerical:* Standard Scaling for Distance and Cyclical Time features.
* **Algorithm:** XGBoost Regressor.
* **Hyperparameters (Optimized):**
    * `n_estimators`: 743
    * `max_depth`: 8
    * `learning_rate`: ~0.04
    * `colsample_bytree`: ~0.65

---

## 📊 Visualizations

### 1. Feature Importance
The model confirmed that **Distance (Manhattan)** is the primary driver, followed heavily by the **Spatial Clusters** and **Hour of Day**, validating the feature engineering strategy.

### 2. Mobility Flow Network
*The project includes code to generate animated flow maps, visualizing how taxi demand shifts from outer boroughs to Manhattan centers throughout the 24-hour cycle. Red nodes represent cluster centroids, and gold arrows represent the volume of flow.*

---

## 🛠️ Project Structure

```bash
├── data/                   # Train and Test CSV files (not included in repo)
├── notebooks/
│   └── NYC_Taxi_EDA_Modeling.ipynb  # Complete Analysis & Pipeline
├── images/                 # Generated plots and flow maps
├── requirements.txt        # Dependencies
└── README.md               # Project Documentation
