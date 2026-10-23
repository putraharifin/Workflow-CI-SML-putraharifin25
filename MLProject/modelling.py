#!/usr/bin/env python3
"""Entry point MLflow Project: retraining RandomForest (class_weight='balanced') dengan mlflow.autolog().

Dijalankan oleh:  mlflow run MLProject/
Menulis run_id.txt (dibaca workflow untuk `mlflow models build-docker`).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("modelling")
TARGET_COL = "Machine failure"


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Salin ai4i2020_preprocessing.csv dari repo Eksperimen_SML_siswa ke folder MLProject/."
        )
    df = pd.read_csv(path)
    
    # Normalisasi nama kolom jika ada perbedaan spasi/underscore dari sistem preprocessing
    df.columns = [col.replace('_', ' ') for col in df.columns]
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ada di {path}. Kolom yang tersedia: {list(df.columns)}")
    y = df[TARGET_COL].astype(int)
    if y.nunique() < 2:
        raise ValueError("Target hanya memiliki satu kelas; tidak bisa dilatih.")
    return df.drop(columns=[TARGET_COL]), y


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retraining RandomForest Predictive Maintenance.")
    p.add_argument("--data-path", type=Path, default=Path("ai4i2020_preprocessing.csv"))
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=20, help="<=0 berarti tanpa batas (None).")
    p.add_argument("--min-samples-split", type=int, default=2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--run-id-file", type=Path, default=Path("run_id.txt"))
    p.add_argument("--experiment-name", default="predictive-maintenance-ci")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    
    X, y = load_dataset(args.data_path)
    
    # Lakukan enkoding sederhana jika kolom 'Type' masih bertipe string/kategori hasil preprocessing
    if 'Type' in X.columns and X['Type'].dtype == 'object':
        X = pd.get_dummies(X, columns=['Type'], drop_first=True)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # Sinkronisasi Tracking URI lokal ke direktori kerja runner
    mlflow.set_tracking_uri("./mlruns")

    # Mencegah tabrakan instans Run ID eksternal bawaan MLflow Project
    active_run = mlflow.active_run()
    if active_run is None:
        mlflow.set_experiment(args.experiment_name)
        run_context = mlflow.start_run()
    else:
        run_context = mlflow.start_run(run_id=active_run.info.run_id)

    mlflow.autolog()  # Aktif penuh melacak skikit-learn
    
    with run_context as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_split=args.min_samples_split,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        LOGGER.info("Test F1=%.4f Recall=%.4f ROC-AUC=%.4f", f1_score(y_test, pred),
                    recall_score(y_test, pred), roc_auc_score(y_test, proba))

        # Menuliskan Run ID ke file teks agar bisa dibaca step build-docker
        args.run_id_file.write_text(run.info.run_id)
        LOGGER.info("Run ID Berhasil Dicatat: %s", run.info.run_id)
        
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        sys.exit(1)
