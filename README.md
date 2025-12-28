# Food Delivery Time Prediction – Production-Ready ETA System

This repository contains a **production-grade implementation** of a real-world  
**Delivery Time (ETA) Prediction System**, inspired by large-scale food delivery platforms such as Swiggy and Zomato.

The objective of this system is to deliver **highly accurate delivery time predictions** using an optimized machine learning pipeline, automated deployment workflow, and scalable cloud infrastructure.

---

## Live Synopsis Dashboard

https://apoorvtechh-synopsis-eta-main-6f3ijc.streamlit.app/

---

## Project Overview

The system predicts **estimated delivery time (ETA)** based on multiple operational and contextual factors, including:

- Delivery partner attributes  
- Restaurant and customer geolocation  
- Order placement, preparation, and pickup timestamps  
- Traffic and weather conditions  
- Vehicle type and order category  
- Distance between restaurant and customer  

This repository contains the **final optimized ML model**, complete preprocessing pipelines, a FastAPI-based inference service, Docker configuration, CI/CD automation, and AWS deployment setup.

---

## Project Repositories

- **Production API Repository:**  
  https://github.com/apoorvtechh/delivery_time_estimator  

- **Experimentation & EDA Repository:**  
  https://github.com/apoorvtechh/Swiggy_project_Experimentation  

---

## Data Preprocessing Pipeline

The production pipeline performs structured and reliable preprocessing to ensure consistency between training and inference.

Key steps include:

- Handling missing, inconsistent, and corrupted values  
- Normalization and transformation of time-based features  
- Feature engineering, including:
  - Haversine distance between restaurant and customer  
  - Order-to-pickup duration  
  - Peak-hour and demand indicators  
- Encoding of categorical variables  
- Feature scaling and normalization  
- Validation of coordinates, delivery partner details, and outlier handling  

This pipeline guarantees **reproducible and stable predictions** in real-time production environments.

---

## Key Insights from EDA (Production-Relevant)

- Distance between restaurant and customer is the strongest predictor of delivery time  
- Order-to-pickup delay significantly impacts overall ETA variance  
- Peak hours introduce non-linear delays due to traffic and demand spikes  
- Weather conditions increase delivery time variance rather than mean ETA  
- Vehicle type influences delivery speed, especially for longer distances  
- Time-based features (hour of day, day of week) improve model stability  

These insights guided feature selection, model design, and deployment decisions.

---

## System Highlights

- End-to-end ML pipeline from raw data to production inference  
- FastAPI-based low-latency prediction service  
- Dockerized deployment for portability and scalability  
- CI/CD-ready structure for automated testing and rollout  
- Cloud-ready architecture suitable for real-world ETA systems  

---

## License

This project is intended for educational, research, and portfolio demonstration purposes.
