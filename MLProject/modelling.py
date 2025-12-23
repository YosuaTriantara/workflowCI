import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.utils import estimator_html_repr

def train(data_file, n_estimators):
    print("Memulai Training Otomatis via MLflow Project")
    
    #SETUP DAGSHUB & MLFLOW 
    DAGSHUB_USERNAME = "yosuatriantara"
    DAGSHUB_REPO_NAME = "SMSML_yosuatriantara"
    token = os.getenv("DAGSHUB_TOKEN")
    
    if token:
        dagshub.auth.add_app_token(token)
        print("DagsHub terautentikasi via token.")

    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment("CI_CD_Churn_Experiment")

    #LOAD DATA
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X = df.drop(columns=['Target_Churn'])
    y = df['Target_Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #TRAINING & LOGGING 
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        print(f"Training RandomForest dengan n_estimators={n_estimators}...")
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "avg_precision": ap
        })
        print(f"Akurasi: {acc} | ROC AUC: {auc}")

        print("logging...")

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig("precision_recall_curve.png")
        mlflow.log_artifact("precision_recall_curve.png")
        plt.close()

        # Estimator HTML Representation
        model_html = estimator_html_repr(model)
        with open("model_structure.html", "w", encoding="utf-8") as f:
            f.write(model_html)
        mlflow.log_artifact("model_structure.html")

        # Dataset Statistics Summary
        X_train.describe().transpose().to_csv("dataset_summary.csv")
        mlflow.log_artifact("dataset_summary.csv")

        # LOGGING MODEL MANUAL
        mlflow.sklearn.log_model(model, "model")

        # EXPORT RUN ID UNTUK GITHUB ACTIONS
        with open("last_run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print(f"Model berhasil disimpan. Run ID: {run.info.run_id}")

        # Cleanup file lokal agar workspace CI/CD tetap bersih
        temp_files = ["confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png", 
                      "model_structure.html", "dataset_summary.csv"]
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="online_retail_customer_churn_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    
    train(args.data_file, args.n_estimators)