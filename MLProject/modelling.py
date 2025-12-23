import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(data_file, n_estimators):
    print("=== Memulai Training Otomatis via MLflow Project ===")
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X = df.drop(columns=['Target_Churn'])
    y = df['Target_Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Setup MLflow
    mlflow.set_tracking_uri("./mlruns") 
    mlflow.set_experiment("CI_CD_Churn_Experiment")

    # 3. Training & Logging
    with mlflow.start_run() as run:
        # Log Parameter
        mlflow.log_param("n_estimators", n_estimators)
        
        # Train Model
        print(f"Training RandomForest dengan n_estimators={n_estimators}...")
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        print(f"Akurasi: {acc}")

        mlflow.sklearn.log_model(model, "model")
        
        with open("last_run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print(f"Model saved in run_id: {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="online_retail_customer_churn_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    
    train(args.data_file, args.n_estimators)