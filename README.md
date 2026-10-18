# Workflow-CI

Retraining otomatis memakai standar MLflow Project, lalu build & push Docker image.

```
.github/workflows/main.yml     # mlflow run MLProject/ -> mlflow models build-docker -> push Docker Hub
MLProject/
├── MLProject                  # entry point
├── conda.yaml                 # Python 3.12.7
├── modelling.py               # RandomForest + mlflow.autolog()
└── ai4i2020_preprocessing.csv # salin dari repo Eksperimen_SML_siswa (hasil preprocessing)
```

GitHub Secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
