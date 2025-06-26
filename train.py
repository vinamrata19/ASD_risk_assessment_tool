import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

# ===============================
# Loading and Preprocessing Dataset
# ===============================
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})

    def convertAge(age):
        if age < 4:
            return 'Toddler'
        elif age < 12:
            return 'Kid'
        elif age < 18:
            return 'Teenager'
        elif age < 40:
            return 'Young'
        else:
            return 'Senior'

    df['ageGroup'] = df['age'].apply(convertAge)

    def add_feature(data):
        data['sum_score'] = data.loc[:, 'A1_Score':'A10_Score'].sum(axis=1)
        data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
        return data

    df = add_feature(df)
    df['age'] = np.log(df['age'])

    def encode_labels(data):
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        return data

    df = encode_labels(df)
    return df

# ===============================
# Preparing Data
# ===============================
def prepare_data(df):
    removal = ['ID', 'age_desc', 'used_app_before', 'austim']
    features = df.drop(removal + ['Class/ASD'], axis=1)
    target = df['Class/ASD']

    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(X_train_imputed, Y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_val_scaled = scaler.transform(X_val_imputed)

    return X_scaled, Y_resampled, X_val_scaled, Y_val, scaler, X_train.columns

# ===============================
# Training and Saving Models with Hyperparameter Tuning
# ===============================
def train_and_save_models(X, Y, X_val, Y_val):
    models = {
        'logistic_regression': (LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10], 'penalty': ['l2']
        }),
        'xgboost': (XGBClassifier(use_label_encoder=False, eval_metric='auc'), {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }),
        'svm': (SVC(probability=True), {
            'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']
        })
    }

    results = {}
    best_model_name = None
    best_val_auc = 0.0

    for name, (base_model, param_grid) in models.items():
        print(f"\n🔍 Tuning {name.upper()}...")
        pipe = Pipeline([('model', base_model)])
        search = RandomizedSearchCV(pipe, {'model__' + k: v for k, v in param_grid.items()},
                                    n_iter=5, scoring='roc_auc', cv=5, random_state=42)
        search.fit(X, Y)

        best_model = search.best_estimator_['model']
        train_pred = best_model.predict(X)
        val_pred = best_model.predict(X_val)
        val_auc = metrics.roc_auc_score(Y_val, val_pred)

        results[name] = {
            'train_auc': metrics.roc_auc_score(Y, train_pred),
            'val_auc': val_auc,
            'train_report': classification_report(Y, train_pred),
            'val_report': classification_report(Y_val, val_pred),
            'model': best_model
        }

        joblib.dump(best_model, f"{name}.joblib")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_name = name

    # Save best model
    joblib.dump(results[best_model_name]['model'], 'best_model.joblib')
    print(f"\n🏆 Best Model: {best_model_name.upper()} (Validation AUC = {best_val_auc:.4f})")
    return results, best_model_name

# ===============================
# Main
# ===============================
def main():
    df = load_and_preprocess_data('autism_dataset.csv')
    X, Y, X_val, Y_val, scaler, feature_columns = prepare_data(df)

    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(feature_columns.tolist(), 'feature_columns.joblib')

    results, best_model_name = train_and_save_models(X, Y, X_val, Y_val)

    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} Model Results:")
        print(f"Training AUC: {model_results['train_auc']:.4f}")
        print(f"Validation AUC: {model_results['val_auc']:.4f}")
        print("Training Report:\n", model_results['train_report'])
        print("Validation Report:\n", model_results['val_report'])

    best_model = results[best_model_name]['model']
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(best_model, X_val, Y_val)
    plt.title(f'Confusion Matrix - {best_model_name.upper()}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("\n✅ Training complete. Best model and scaler saved.")

if __name__ == '__main__':
    main()
