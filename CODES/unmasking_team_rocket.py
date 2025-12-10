import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

class TeamRocketDetector:
    def __init__(self, model_path='xgb_model.pkl', features_path='model_features.pkl'):
        """
        Initializes the detector with paths for saving/loading artifacts.
        """
        self.model = XGBClassifier(
            n_estimators=75, 
            max_depth=10, 
            learning_rate=0.4, 
            random_state=42
        )
        self.model_path = model_path
        self.features_path = features_path
        self.features_list = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, url):
        """
        Loads data from the source and handles basic indexing.
        """
        print(f"Loading data from {url}...")
        self.data = pd.read_csv(url)
        # Adjusting index to 1-based as per original script
        self.data.index = range(1, len(self.data) + 1)
        print(f"Data loaded successfully. Shape: {self.data.shape}")

    def preprocess(self):
        """
        Separates labeled data, handles encoding, and aligns features.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1. Segregate Labeled Data (Target is not null)
        df_label = self.data[self.data['Team Rocket'].notnull()].copy()
        
        # 2. Separate Target and Features
        y = df_label['Team Rocket'].map({'Yes': 1, 'No': 0})
        
        # 3. Split Categorical and Numerical
        # Note: We drop target from features
        df_cat = df_label.select_dtypes(include='object').drop('Team Rocket', axis=1)
        df_num = df_label.select_dtypes(exclude='object')

        # 4. Encoding (Get Dummies)
        df_encoded = pd.get_dummies(df_cat, drop_first=True)
        
        # 5. Concatenate
        X = pd.concat([df_num, df_encoded], axis=1)

        # --- CRITICAL STEP FOR PRODUCTION ---
        # Save the column names. If new data comes in, we must ensure it has 
        # exactly these columns in this order.
        self.features_list = X.columns.tolist()
        joblib.dump(self.features_list, self.features_path)
        print(f"Feature list saved to {self.features_path}")

        # 6. Train/Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.125, random_state=42
        )
        print("Preprocessing complete. Data split into Train and Test sets.")

    def train(self):
        """
        Fits the XGBoost classifier.
        """
        print("Training XGBoost Model...")
        self.model.fit(self.X_train, self.y_train)
        print("Training complete.")

    def evaluate(self, threshold=0.45):
        """
        Evaluates the model using Accuracy and ROC-AUC.
        Custom thresholding is applied for decision making.
        """
        print(f"\n--- Evaluation (Threshold: {threshold}) ---")
        
        # Get probabilities
        y_proba = self.model.predict_proba(self.X_test)
        
        # Apply custom threshold (Aggressive detection)
        y_pred = (y_proba[:, 1] >= threshold).astype(int)

        # Metrics
        acc = accuracy_score(self.y_test, y_pred)
        roc = roc_auc_score(self.y_test, y_pred)
        
        print(f"Accuracy Score: {acc:.4f}")
        print(f"ROC-AUC Score:  {roc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return acc, roc

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

# --- Usage Example ---

if __name__ == "__main__":
    # 1. Initialize
    detector = TeamRocketDetector()

    # 2. Load
    url = 'https://docs.google.com/spreadsheets/d/1o_Msk7Lw8HQiHVKmR2GlnV2FaxBMk96gNpgKw5Un3XI/export?format=csv'
    detector.load_data(url)

    # 3. Preprocess
    detector.preprocess()

    # 4. Train
    detector.train()

    # 5. Evaluate
    detector.evaluate(threshold=0.45)

    # 6. Save
    detector.save_model()