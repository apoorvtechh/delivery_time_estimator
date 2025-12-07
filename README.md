# 🛵 Food Delivery Time Prediction – Production-Ready ETA System

This repository contains the **production-grade implementation** of a real-world  
**Delivery Time (ETA) Prediction System**, inspired by platforms like **Swiggy** and **Zomato**.

The goal of this system is to provide **highly accurate delivery time predictions** using an optimized ML pipeline, automated deployment workflow, and scalable cloud infrastructure.

---

## 📌 Live Synopsis Dashboard  
🔗 https://apoorvtechh-synopsis-eta-main-6f3ijc.streamlit.app/

---

# 🚀 Project Overview

The system predicts **how long a delivery will take** based on key operational and contextual features:

- 👤 Delivery partner details  
- 🍽 Restaurant & 📍 customer locations  
- 🕒 Order and pickup timestamps  
- 🌦 Weather and 🚦 traffic conditions  
- 🛵 Vehicle type & order category  
- 🧭 Distance between restaurant → customer  

This repository includes the **final optimized ML model**, preprocessing pipelines, FastAPI backend, Docker setup, CI/CD automation, and AWS deployment infrastructure.

👉 **Production API Repository:**  
🔗 https://github.com/apoorvtechh/delivery_time_estimator  

👉 **Experimentation Repository (EDA + Research):**  
🔗 https://github.com/apoorvtechh/Swiggy_project_Experimentation  

---

# 🧹 Data Preprocessing Pipeline

The production pipeline performs structured, reliable preprocessing:

- Handling missing and corrupted values  
- Converting and normalizing time-based features  
- Feature engineering including:  
  - **Haversine distance**  
  - **Order-to-pickup duration**  
  - **Peak hour indicators**  
- Encoding categorical fields  
- Scaling & normalization for model readiness  
- Validation of coordinates, rider details, and outliers  

This preprocessing flow ensures **consistent, reproducible performance** during real-time inference.

---

# 📊 Key Insights from EDA (Summarized for Production)

Insights leveraged during modeling:

- Relationship between traffic density and delivery speed  
- Weather impact on ETA variability  
- Patterns in partner efficiency and route behavior  
- City-wise delivery performance differences  
- Target variable distribution shaping  
- Correlation-driven feature selection  

These insights informed **final feature engineering and model choices**.

---

# 🤖 Model Architecture (Production Version)

Multiple ML models were benchmarked, and the final system uses a **Weighted Ensemble** for best real-world performance:

### Models Selected:
- **LightGBM**  
- **CatBoost**  

Additional models evaluated during experimentation:
- XGBoost  
- SVM  
- Random Forest  

---

# 📈 Model Performance (Final Metrics)

The production ensemble achieves:

- **MAE ≈ 3.01 minutes**  
- **R² ≈ 0.84**  

This combination balances **accuracy, speed, and stability**, making it ideal for real-time prediction scenarios.

---

# ⚙️ Production Deployment Stack

The system is deployed using a scalable ML engineering stack:

- **FastAPI** for real-time inference  
- **Docker** for containerized execution  
- **GitHub Actions** for CI/CD automation  
- **AWS EC2 + ECR + S3** for cloud hosting  
- **AWS Auto Scaling + Application Load Balancer (ALB)** for high availability  
- Load-tested with **100k+ requests** ensuring reliable scaling  

---

If you want, I can help you add:

✔ Architecture diagram  
✔ Demo GIF  
✔ API documentation section  
✔ Project badge section (shields.io)


