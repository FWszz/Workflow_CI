import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data(path='titanic_preprocessing.csv'):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators',      type=int, default=200)
    parser.add_argument('--max_depth',         type=int, default=8)
    parser.add_argument('--min_samples_split', type=int, default=4)
    parser.add_argument('--min_samples_leaf',  type=int, default=2)
    args = parser.parse_args()

    # Bersihkan environment MLflow yang di-inject GitHub Actions
    os.environ.pop('MLFLOW_RUN_ID', None)
    os.environ.pop('MLFLOW_EXPERIMENT_ID', None)
    mlflow.end_run()

    mlflow.set_experiment('Titanic-Survival-Prediction')

    # Aktifkan autolog — otomatis mencatat params, metrics, dan model artifact
    mlflow.sklearn.autolog()

    data_path = os.path.join(os.path.dirname(__file__), 'titanic_preprocessing.csv')
    df = load_data(data_path)

    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        'n_estimators':      args.n_estimators,
        'max_depth':         args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf':  args.min_samples_leaf,
        'random_state':      42,
    }

    with mlflow.start_run(run_name='RandomForest_CI', nested=False):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n[RandomForest_CI]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred))
        print("\n✅ Training selesai dan dicatat di MLflow!")
