# 🚀 Delivery Time Estimator — Production-Grade ML System

This repository contains the complete **production implementation** of a machine-learning powered  
**Food Delivery Time (ETA) Prediction System**, built with:

- **FastAPI** (real-time inference)
- **AWS Auto Scaling Group (ASG)** (self-scaling backend)
- **Application Load Balancer (ALB)** (traffic distribution)
- **Docker + Amazon ECR** (containerized deployment)
- **DVC** (ML pipeline & data versioning)
- **MLflow** (experiment tracking + model registry)
- **CI/CD (GitHub Actions)** (automated build & deploy)

This is the final, scalable, cloud-ready version of the project.

---

## 📦 Project Summary

The goal of this system is to predict **delivery time (ETA)** for food delivery platforms like Swiggy/Zomato based on:

- Restaurant & customer GPS coordinates  
- Delivery partner details  
- Weather and traffic conditions  
- Order timestamps & pickup delays  
- Order type & vehicle type  
- Engineered features like distance, time delta, etc.

The system is designed for **real-time prediction at scale**, with AWS Auto Scaling ensuring reliability under heavy traffic.

---

## 🧠 Machine Learning Pipeline

The ML workflow is fully managed using **DVC** and **MLflow**.

### 🔹 Pipeline Stages
- Data cleaning & preprocessing  
- Feature engineering (Haversine distance, time deltas, LOF outlier removal)  
- Exploratory Data Analysis  
- Model training & comparison  
- Hyperparameter tuning  
- Weighted ensemble creation  
- Model evaluation  
- Packaging model for deployment  

### 🔹 Best Performing Model
A **weighted ensemble**:

- **LightGBM** → 60%  
- **CatBoost** → 40%  

Saved and served via **MLflow Model Registry**.

---

## ⚙️ FastAPI Backend (Real-Time Inference)

The inference API:

- Loads the latest model from MLflow Registry  
- Validates JSON input with **Pydantic**  
- Applies preprocessing pipeline  
- Runs inference in milliseconds  
- Returns predicted ETA  




End-to-end machine learning pipeline for delivery ETA prediction using DVC, MLflow, and modular data workflows.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
