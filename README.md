# Automated MLOps Pipeline for Fraud Detection 🏭

[![CI/CD Pipeline](https://github.com/amirulhazym/mlops-automated-pipeline/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/amirulhazym/mlops-automated-pipeline/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

This repository documents the construction of a comprehensive, production-style MLOps pipeline. The project demonstrates a full-cycle approach to machine learning systems, encompassing scalable data engineering, multi-framework model experimentation, automated CI/CD, dual-target deployment (serverless and container orchestration), and a multi-faceted monitoring strategy.

The primary goal is to build a robust, reproducible, and automated system for a real-world fraud detection use case, leveraging a stack of industry-standard, open-source tools and cloud services, with a strong emphasis on professional engineering practices.

## 🏗️ System Architecture

This diagram illustrates the complete workflow of the MLOps pipeline, from data acquisition to monitoring.

```mermaid
graph TD
    subgraph "A. Local Development Environment 💻"
        direction LR
        subgraph "1. Data Engineering & Versioning"
            A[Hugging Face Dataset] --> B(script: acquire_dataset.py);
            B -- Generates --> C[Raw Data .csv];
            C --> D(script: preprocess_with_spark.py);
            D -- Creates --> E[Processed Data .parquet];
            E -- "dvc add" --> F{DVC};
            F -- "dvc push" --> G[("AWS S3 Remote")];
        end

        subgraph "2. Model Experimentation"
            H("script: train_model.py") -- Uses --> E;
            H -- Trains --> I1[XGBoost Model];
            H -- Trains --> I2[TensorFlow Model];
            I1 & I2 -- "mlflow.log_model()" --> J[("MLflow Tracking Server")];
            H -- "mlflow.log_params/metrics()" --> J;
        end

        subgraph "3. Source Code & Configuration"
            K1[Python Scripts <br> /src, /scripts];
            K2[DVC Metafile <br> .dvc];
            F -- Creates --> K2;
            K3[Dockerfiles];
            K4[Kubernetes Manifests <br> /k8s];
            K5[CI/CD Workflow <br> .github/workflows];
        end
    end

    "A. Local Development Environment 💻" -- "git commit & push" --> L{GitHub Repository 🐙};

    subgraph "B. CI/CD Automation Pipeline (GitHub Actions) 🚀"
        L -- "Triggers Workflow <br> on push to main" --> M;
        subgraph "CI: Quality Gates"
            M(1. Lint & Test) -- "on success" --> N(2. Build Training Docker Image);
        end
        subgraph "CD: Automated Deployment"
            N -- "on success" --> O(3. Deploy P1 API to AWS);
        end
    end

    subgraph "C. Deployment Targets ☁️"
        subgraph "Serverless (Deployed by CD)"
            P[("AWS Lambda & API Gateway")];
        end
        subgraph "Container Orchestration (Deployed Manually)"
            Q{Kubernetes Cluster <br> (Minikube)};
            R[P1 API Docker Image] -- "minikube image load" --> Q;
            K4 -- "kubectl apply" --> Q;
        end
    end

    O -- "via AWS SAM" --> P;

    subgraph "D. Monitoring & Observability 📊"
        subgraph "Application Performance Monitoring (APM)"
            Q -- "Exposes /metrics" --> S(Prometheus);
            S -- "Data Source for" --> T[/Grafana Dashboard/];
        end
        subgraph "ML Model Quality Monitoring"
            U(Evidently AI Script) -- Uses --> E & J;
            U -- Generates --> V[Data Drift & Performance Reports];
        end
        subgraph "Real-Time POC"
            W(Kafka Producer) -- "Sends Events" --> X[("Kafka Topic")];
            X -- "Streams to" --> Y(Kafka Consumer);
            Y -- "Calls API on" --> Q;
        end
    end

    %% Styling
    style G fill:#FF9900,stroke:#333,stroke-width:2px
    style J fill:#C8E6C9,stroke:#333,stroke-width:2px
    style L fill:#181717,color:#fff,stroke-width:2px,stroke:#555
    style P fill:#FF9900,stroke:#333,stroke-width:2px
    style Q fill:#326CE5,stroke:#333,stroke-width:2px,color:#fff
    style X fill:#FFF9C4,stroke:#333,stroke-width:2px
    style T fill:#E6522C,color:#fff
```

## ⭐ Core Features

- **Scalable Data Processing:** A configurable Apache Spark pipeline processes a 6.3 million row dataset.
- **Comprehensive Version Control:** A unified strategy using Git for code, DVC for large data artifacts (with AWS S3 remote), and MLflow for experiment and model versioning.
- **Multi-Model Experimentation:** A flexible training script that supports a "bake-off" between XGBoost and a TensorFlow neural network, tracked in MLflow.
- **Automated CI/CD Pipeline:** A robust GitHub Actions pipeline that automatically lints, tests, builds Docker images, and deploys the inference API to AWS Lambda via SAM and a secure OIDC connection.
- **Dual-Target Deployment:** The same inference API is deployed to two modern environments:
  - **Serverless:** AWS Lambda & API Gateway (via automated CD).
  - **Container Orchestration:** A local Kubernetes cluster managed by Minikube.
- **Real-Time Architecture POC:** A Proof-of-Concept using Apache Kafka to simulate a real-time stream of transactions processed by the API.
- **Multi-Faceted Monitoring Stack:**
  - **ML Quality Monitoring:** Evidently AI generates data drift and model performance reports.
  - **Application Performance Monitoring (APM):** A live stack using Prometheus for metrics collection and Grafana for real-time visualization of API latency and request rates.

## 🛠️ Technology Stack

| Category | Technologies Used |
|----------|------------------|
| **Data & ML** | Python 3.11, Pandas, Scikit-learn, Apache Spark (PySpark), XGBoost, TensorFlow (Keras) |
| **MLOps & Automation** | MLflow, DVC, GitHub Actions (CI/CD), AWS SAM, Pytest, Flake8 |
| **Deployment & Cloud** | Docker, Docker Compose, Kubernetes (Minikube), AWS S3, AWS Lambda, AWS API Gateway, AWS ECR, IAM |
| **Monitoring & Streaming** | Apache Kafka, Prometheus, Grafana, Evidently AI |

## 📁 Project Structure

```
mlops-automated-pipeline/
├── .github/workflows/          # GitHub Actions CI/CD pipeline (ci_pipeline.yml)
├── data/                       # Managed by scripts and DVC
├── docs/                       # Project documentation and monitoring reports
├── k8s/                        # Kubernetes manifest files (deployment.yaml, service.yaml)
├── mlruns/                     # Local MLflow experiment data (.gitignore'd)
├── notebooks/                  # Jupyter notebooks for EDA
├── scripts/                    # Helper scripts (data acquisition, monitoring)
├── src/                        # Main application source code
│   ├── data_engineering/       # Spark preprocessing script
│   ├── p1_api_deployment/      # Source files for the P1 API deployment
│   └── training/               # Model training script
├── .dockerignore
├── .flake8
├── .gitignore
├── docker-compose.yml          # Defines local Kafka & Zookeeper services
├── requirements.txt            # Core application dependencies
└── requirements-dev.txt        # Development and testing dependencies
```

## 🚀 Local Setup & Usage

### Prerequisites

- Git, Python 3.11, Docker Desktop
- An AWS account with the AWS CLI and SAM CLI installed and configured.
- Minikube and kubectl.
- Correctly configured Java (JDK 17) and Hadoop environment variables for Spark on Windows.

### Installation & Workflow

1. **Clone & Install:**
   ```bash
   git clone https://github.com/amirulhazym/mlops-automated-pipeline.git
   cd mlops-automated-pipeline
   python -m venv p2env
   .\p2env\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Run the Pipeline Stages:**
   ```bash
   # 1. Acquire Data
   python scripts/acquire_dataset.py --sample_size 1000000 # For dev
   python scripts/acquire_dataset.py # For full data

   # 2. Preprocess with Spark
   python src/data_engineering/preprocess_with_spark.py --input_file data/raw_data/full_fraud_data.csv --output_suffix full

   # 3. Version Data with DVC
   dvc add data/processed/engineered_features_full
   git commit -m "Data: Version full processed dataset"
   dvc push

   # 4. Train Models (start 'mlflow ui' in another terminal)
   python src/training/train_model.py --model_type xgboost --data_version full --run_name "XGBoost_Full_v1"

   # 5. Deploy to Kubernetes
   minikube start --profile mlops-cluster
   docker build -t p1-fraud-api:final -f src/p1_api_deployment/Dockerfile src/p1_api_deployment/
   minikube image load p1-fraud-api:final --profile mlops-cluster
   kubectl apply -f k8s/
   minikube service p1-fraud-api-service --profile mlops-cluster
   ```

## 💡 Key Challenges & Learnings

This project was a deep dive into the practical realities of building and debugging a modern MLOps system.

- **Complex Environment Configuration:** The initial setup of a local Spark environment on Windows was a significant challenge, requiring systematic debugging of Java/PySpark version incompatibilities, HADOOP_HOME dependencies, and Python worker pathing.
- **CI/CD Deployment Failures:** The automated deployment pipeline failed for multiple real-world reasons, providing critical lessons in debugging IAM permissions (missing ECR access), deprecated SAM CLI flags, and the intricate workflow of sam build vs. sam deploy for containerized applications.
- **Kubernetes Debugging Cycle:** Deploying to Kubernetes was a multi-stage debugging process. I solved ImagePullBackOff errors by understanding the context separation between the host and Minikube Docker daemons, solved a CrashLoopBackOff by re-engineering the Docker CMD instruction, and finally stabilized the pod by implementing proper liveness and readiness probes.
- **Storage Management:** I proactively identified and solved a critical low-storage issue on my C: drive by diagnosing Docker Desktop's virtual disk behavior and successfully migrating its entire WSL data root to a larger drive using wsl --export/--import.

## 🔮 Future Enhancements

- Run data processing and training jobs on managed cloud services (e.g., AWS EMR, SageMaker).
- Implement a centralized, cloud-hosted MLflow server.
- Harden the CI/CD pipeline with GitOps principles (e.g., ArgoCD).
- Configure proactive alerting in Grafana via Alertmanager.

## 👤 Author

**Amirulhazym**

- LinkedIn: [linkedin.com/in/amirulhazym](https://linkedin.com/in/amirulhazym)
- GitHub: [github.com/amirulhazym](https://github.com/amirulhazym)
- Portfolio: [amirulhazym.framer.ai](https://amirulhazym.framer.ai)
