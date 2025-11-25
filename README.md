📦 E-commerce Sales Forecasting Pipeline
End-to-End MLOps Project 

🔗 Live Demo (Front-End Web App)
👉 https://huggingface.co/spaces/zkmine/Ecommerce-Prediction

🔗 Cloud Run API Endpoint
👉 https://ecommerce-ml-346297770564.us-east1.run.app/predict

📁 Project Overview

This project implements a complete machine learning lifecycle for e-commerce sales forecasting, including:

Dataset preprocessing

Model training and experiment tracking

Feature schema and model artifact storage

FastAPI-based model serving

Dockerized deployment to Google Cloud Run

Interactive front-end built on Gradio, deployed on Hugging Face Spaces

The pipeline is fully automated, modular, and cloud-ready.

🚀 Features
✔ End-to-End ML pipeline

Preprocessing

Feature engineering

Model training (Linear Regression / others optional)

Evaluation

Artifacts export (joblib + schema JSON)

✔ Model Serving (FastAPI)

/health endpoint

/predict endpoint

/schema endpoint

Supports dynamic feature validation

Uses Pydantic for robust input checking

✔ Deployment

Dockerfile included → runs end-to-end

Deployable locally or on Google Cloud Run

Hugging Face “Docker template” supported via Spaces

✔ Experiment Tracking

MLflow OR Weights & Biases logging included (choose one)

✔ Front-End Web App

Built with Gradio

Dynamic UI based on schema

Numeric + Binary Yes/No UI

Styled prediction output

Calls Cloud Run API directly
