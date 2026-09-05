# E-Commerce Revenue Analytics & Customer Segmentation

An end-to-end data science and machine learning project analyzing **1.1 million** transactional e-commerce order records across 20 metropolitan and regional markets in India (January 2024 – August 2026).

---

## 📌 Project Overview
The objective of this project is to examine revenue attribution, evaluate geographical demand dispersion, segment city markets via unsupervised learning, and build predictive regression pipelines to forecast order revenue.

* **Dataset Volume**: 1,100,000 unique records (0 missing values, 0 duplicate rows)
* **Time Span**: January 1, 2024 – August 16, 2026
* **Catalog Scope**: 20 distinct products spanning 9 retail categories across 20 cities

---

## 🚀 Key Business Insights
1. **Category Revenue Asymmetry**:
   * **Electronics** constitutes **₹25.17 Billion** (~75.6%) of total gross platform revenue, driven by high average unit pricing (~₹30,411).
   * **Furniture** ranks second (~₹2.88B), while **Books** accounts for the lowest revenue contribution (~₹111.4M).
2. **Decentralized Regional Purchasing Power**:
   * Tier-2 and emerging commercial hubs (e.g., **Kochi**, **Bhopal**, **Lucknow**) match or outperform Tier-1 metros (e.g., **Delhi**, **Mumbai**, **Bengaluru**) in Average Order Value (AOV ~₹30,500+).
3. **Inelastic Basket Volume**:
   * Order quantity remains stable across all categories with an Interquartile Range (IQR) of 2 to 4 units (median: 3 units). Order revenue variance is primarily driven by unit price rather than basket size.

---

## 🧠 Machine Learning Architecture

### 1. Market Segmentation (K-Means Clustering)
* **Methodology**: Standard feature normalization (`StandardScaler`) across total revenue, order count, total volume, and average order value.
* **Optimal Clusters ($k=3$)**:
  * **Tier 1 (High Revenue)**: High-AOV regional and metro hubs (e.g., Kochi, Surat, Hyderabad, Delhi).
  * **Tier 2 (High Volume)**: Consistent bulk-order regional demand (e.g., Bhopal, Lucknow, Mumbai, Pune).
  * **Tier 3 (Growth/Emerging)**: Developing catalog nodes (e.g., Bengaluru, Noida, Jaipur, Kolkata).
* **Validation Metrics**:
  * **Silhouette Score**: `0.3431`
  * **Davies-Bouldin Index**: `0.8627` (< 1.0 indicates well-separated clusters)

### 2. Transaction Revenue Prediction (Random Forest Regressor)
* **Predictor Variables**: `Product_Code`, `Category_Code`, `City_Code`, `Quantity`, `Month`, `DayofWeek`, `Day`
* **Train/Test Split**: 80,000 training samples / 20,000 test samples (random state 42)
* **Model Evaluation**:
  * **R² Score**: `0.8874` (~88.7% variance explained)
  * **RMSE**: `₹20,106.40`
  * **MAE**: `₹8,901.60`
* **Top Revenue Drivers**:
  1. `Product_Code` (~53.4% importance)
  2. `Quantity` (~20.7% importance)
  3. `Category_Code` (~15.9% importance)

---

## 🛠️ Tech Stack
* **Language**: Python 3.10+
* **Data Processing**: pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
* **Machine Learning**: scikit-learn
