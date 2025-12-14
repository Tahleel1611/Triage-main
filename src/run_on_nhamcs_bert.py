import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def main():
    # 1. Load Data
    csv_path = 'data/nhamcs_combined.csv'
    bert_path = 'data/nhamcs_bert_features.npy'
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loading BERT features from {bert_path}...")
    try:
        bert_features = np.load(bert_path)
    except FileNotFoundError:
        print("BERT features not found. Please run src/bert_feature_extraction.py first.")
        return

    # Align Data (Truncate DF to match BERT features)
    # Note: We assume the order is preserved and 1:1 mapping
    n_samples = bert_features.shape[0]
    print(f"BERT features shape: {bert_features.shape}")
    print(f"Original DF shape: {df.shape}")
    
    if n_samples < df.shape[0]:
        print(f"Truncating dataframe to first {n_samples} rows to match BERT features.")
        df = df.iloc[:n_samples]
    
    # 2. Preprocessing
    # Drop rows with missing target
    # IMPORTANT: If we drop rows from DF, we must drop corresponding rows from BERT features.
    # But BERT features are numpy array.
    # Let's find indices to keep.
    
    # Check for missing target
    missing_target_mask = df['ESI'].isna()
    if missing_target_mask.any():
        print(f"Dropping {missing_target_mask.sum()} rows with missing ESI...")
        df = df.dropna(subset=['ESI'])
        bert_features = bert_features[~missing_target_mask]
        
    # Features
    numeric_features = ['Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat', 'PainScale']
    categorical_features = ['ArrivalMode']
    # Text feature is replaced by BERT embeddings
    
    X_df = df[numeric_features + categorical_features]
    y = df['ESI']

    # Encode Target (1-5 -> 0-4)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print(f"Final Data Shape: {X_df.shape}")
    print(f"Final BERT Shape: {bert_features.shape}")

    # 3. Define Pipelines
    
    # Numeric Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessor for structured data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # We need to combine structured data processing with BERT features.
    # Since BERT features are already numeric and don't need scaling (usually), we can just concatenate.
    # But we need a custom transformer or do it outside the pipeline.
    # To keep it clean, let's process X_df first, then concat.
    
    print("Preprocessing structured data...")
    X_structured = preprocessor.fit_transform(X_df)
    
    # Concatenate
    # X_structured might be sparse if OneHot is sparse.
    if hasattr(X_structured, "toarray"):
        X_structured = X_structured.toarray()
        
    X_combined = np.hstack([X_structured, bert_features])
    print(f"Combined Feature Shape: {X_combined.shape}")

    # 4. Model Definition (Stacking)
    # Base Learners
    # Reduced estimators for speed in demo
    lgbm = LGBMClassifier(random_state=42, verbose=-1, n_estimators=50) 
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    gb = GradientBoostingClassifier(n_estimators=20, random_state=42)

    estimators = [
        ('lgbm', lgbm),
        ('rf', rf),
        ('gb', gb)
    ]

    # Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced'),
        n_jobs=1,
        cv=3
    )

    # Full Pipeline with SMOTE
    # Since we already preprocessed, we just need SMOTE and Classifier
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', stacking_clf)
    ])

    # 5. Evaluation
    print("\nStarting Cross-Validation (3-fold)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_combined, y, cv=cv, scoring='accuracy', n_jobs=1, verbose=2)

    print(f"\nMean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 6. Train Final Model
    print("\nTraining final model on full dataset...")
    pipeline.fit(X_combined, y)
    
    # Save
    model_path = 'data/nhamcs_bert_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
