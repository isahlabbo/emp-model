import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Data Preparation Function
def prepare_data(data_size=1000, num_features=10):
    data = pd.read_csv('data.csv')  # Replace with actual data file
    X = data.iloc[:, :-1]          # Features
    y = data.iloc[:, -1]           # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Training Function
def train_models(X_train, y_train):
    print("\nTraining Models:")
    progress = tqdm(total=3, desc="Progress")

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    progress.update(1)

    # LSTM
    X_train_lstm = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
    progress.update(1)

    # SVM
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    progress.update(1)

    progress.close()
    return rf_model, lstm_model, svm_model, scaler

# Testing Function
def test_models(models, scaler, X_test):
    rf_model, lstm_model, svm_model = models
    print("\nTesting Models:")
    progress = tqdm(total=3, desc="Progress")

    # Standardize test data
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    progress.update(1)

    # LSTM
    X_test_lstm = X_test_scaled.reshape(-1, 1, X_test_scaled.shape[1])
    lstm_prob = lstm_model.predict(X_test_lstm).flatten()
    progress.update(1)

    # SVM
    svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    progress.update(1)

    progress.close()
    return rf_prob, lstm_prob, svm_prob

# Evaluation Function
def evaluate_models(y_test, rf_prob, lstm_prob, svm_prob):
    print("\nEvaluating Hybrid Model:")
    progress = tqdm(total=1, desc="Progress")

    # Hybrid Model
    hybrid_prob = (rf_prob + lstm_prob + svm_prob) / 3
    hybrid_pred = (hybrid_prob > 0.5).astype(int)

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, hybrid_pred),
        "Precision": precision_score(y_test, hybrid_pred),
        "Recall": recall_score(y_test, hybrid_pred),
        "F1-Score": f1_score(y_test, hybrid_pred),
        "ROC-AUC": roc_auc_score(y_test, hybrid_pred)
    }

    progress.update(1)
    progress.close()
    return metrics

# Main Function to Demonstrate Interaction
def main():
    # Data Preparation
    X_train, X_test, y_train, y_test = prepare_data()

    # Train Models
    rf_model, lstm_model, svm_model, scaler = train_models(X_train, y_train)

    # Test Models
    rf_prob, lstm_prob, svm_prob = test_models((rf_model, lstm_model, svm_model), scaler, X_test)

    # Evaluate Models
    metrics = evaluate_models(y_test, rf_prob, lstm_prob, svm_prob)

    # Display Results
    print("\nHybrid Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
