# Kidney Disease Classification End-to-End Deep Learning Project

An end-to-end Machine Learning Operations (MLOps) project that classifies CT scan images into **Normal** or **Tumor/Disease** categories using Convolutional Neural Networks (CNN) with Transfer Learning (VGG16), MLflow experiment tracking, DVC pipeline automation, and automated deployment.

## Workflow Pipeline
1. Update `config.yaml`
2. Update `params.yaml`
3. Update the entity (`src/cnnClassifier/entity/config_entity.py`)
4. Update configuration manager (`src/cnnClassifier/config/configuration.py`)
5. Update components (`src/cnnClassifier/components/`)
6. Update pipeline (`src/cnnClassifier/pipeline/`)
7. Update `main.py`
8. Update `dvc.yaml`
9. Run `app.py` for web application inference

## Technologies Used
- **Deep Learning**: TensorFlow, Keras, VGG16
- **MLOps**: MLflow, DVC, DagsHub
- **Web Interface**: Flask, HTML5, Modern CSS Glassmorphism
- **DevOps**: Docker, GitHub Actions, AWS CI/CD
