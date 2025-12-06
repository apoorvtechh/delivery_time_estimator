# 🛵 Food Delivery Time Prediction – Experimentation (ML Research Repo)

This repository contains the complete **experimentation workflow** for building a real-world  
**Delivery Time (ETA) Prediction System**, inspired by platforms like **Swiggy** and **Zomato**.

The purpose of this repo is to perform **deep data analysis**, **feature engineering**, and **model experimentation** before integrating the best-performing model into the production pipeline.

**Synopsis**
https://apoorvtechh-synopsis-eta-main-6f3ijc.streamlit.app/
---

# 🚀 Project Overview

The objective of this project is to accurately predict **how long a delivery will take**, based on:

- 👤 Delivery partner details  
- 🍽 Restaurant & 📍 customer locations  
- 🕒 Order & pickup timestamps  
- 🌦 Weather and 🚦 traffic conditions  
- 🛵 Vehicle type & order type  
- 🧭 Distance between restaurant → customer  

This repo includes **all Jupyter notebooks, EDA, model experiments, and preprocessing steps** used during ML research.

👉 **Production API Repository:**  
🔗 https://github.com/apoorvtechh/delivery_time_estimator  


---

# 🧹 Data Preprocessing & Cleaning

### Key preprocessing steps performed:

- Handling missing or corrupted values  
- Normalizing & converting time-based features  
- Creating engineered features like:  
  - **Haversine Distance**  
  - **Order-to-pickup duration**  
  - **Peak hour indicators**  
- Encoding all categorical columns  
- Scaling + normalization for ML input  
- Detecting abnormalities (invalid coordinates, illegal rider ages, synthetic entries)

---

# 📊 Exploratory Data Analysis (EDA)

This repo includes detailed EDA to understand Swiggy/Zomato-style delivery patterns:

- Delivery partner behavior analysis  
- Impact of **traffic density** on delivery speed  
- Influence of **weather** on ETA  
- Understanding city-wise differences  
- Distribution of target variable (Time Taken)  
- Missing data pattern heatmaps  
- Correlation analysis across features  

Visualizations helped shape better modeling decisions & feature engineering.

---

# 🧪 Model Experimentation

Multiple machine learning models were trained, evaluated, and compared, including:

### 🤖 ML Models Tested

- **LightGBM**  
- **CatBoost**  
- **Support Vector Machine (SVM)**  
- **XGBoost Regressor**  
- **Random Forest Regressor**  

### 📈 Metrics Evaluated

Each model was compared on:

- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **R² Score**  

These experiments helped determine the top-performing models, which were later fine-tuned using Optuna and deployed as a **weighted ensemble** in the final production system.

---

# 📂 Repository Purpose

This repo serves as:

✔ A **sandbox** for experimentation  
✔ A record of all **EDA, transformations, and models tried**  
✔ A complementary research repo to the final deployed system  
✔ An essential part of the **ML lifecycle** before deployment  

For actual production code, API development, CI/CD, Docker deployment, and AWS scaling setup, please refer to the final repo below:

👉 **Production Deployment Repo:**  
https://github.com/apoorvtechh/delivery_time_estimator  

---

# 👨‍💻 Author  
**Apoorv Gupta**  
📧 Email: **apoorvtechh@gmail.com**  
🐙 GitHub: https://github.com/apoorvtechh  

---


