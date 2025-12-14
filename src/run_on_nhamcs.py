import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def main():
    # 1. Load Data
    data_path = 'data/nhamcs_combined.csv'
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Combined CSV not found. Please run src/merge_years.py first.")
        return

    # 2. Preprocessing
    # Drop rows with missing target
    df = df.dropna(subset=['ESI'])
    
    # Features
    numeric_features = ['Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat', 'PainScale']
    categorical_features = ['ArrivalMode']
    text_feature = 'Chief_complain'
    target = 'ESI'

    X = df[numeric_features + categorical_features + [text_feature]]
    y = df[target]

    # Encode Target (1-5 -> 0-4)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print(f"Data Shape: {X.shape}")
    print("Class Distribution:\n", pd.Series(y).value_counts())

    # 3. Define Pipelines
    
    # Numeric Pipeline: Impute missing vitals with Median -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline: Impute -> OneHot
    from sklearn.preprocessing import OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Text Pipeline: TF-IDF on Diagnosis Codes
    # We use TF-IDF instead of BERT here for speed on the larger dataset, 
    # and because codes (e.g. "E119") are discrete tokens, not natural language sentences.
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=1000, token_pattern=r'[A-Z]\d{2,4}', lowercase=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ])

    # 4. Model Definition (Stacking)
    
    # Base Learners
    lgbm = LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42)

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
    # Note: SMOTE is applied only on training data during CV
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', stacking_clf)
    ])

    # 5. Evaluation
    print("\nStarting Cross-Validation (3-fold)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1, verbose=2)

    print(f"\nMean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 6. Train Final Model
    print("\nTraining final model on full dataset...")
    pipeline.fit(X, y)
    
    # Save
    model_path = 'data/nhamcs_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
