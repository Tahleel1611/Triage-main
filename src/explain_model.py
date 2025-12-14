import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

def get_feature_names(column_transformer):
    """Get feature names from all transformers."""
    output_features = []

    for name, pipe, features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(pipe, 'get_feature_names_out'):
            # For newer sklearn versions
            output_features.extend(pipe.get_feature_names_out(features))
        elif hasattr(pipe, 'get_feature_names'):
            # For older sklearn versions
            output_features.extend(pipe.get_feature_names(features))
        else:
            # If no method, just use original names (e.g. for 'passthrough')
            output_features.extend(features)
            
    return output_features

def main():
    # 1. Load Model and Data
    model_path = 'data/nhamcs_model.joblib'
    data_path = 'data/nhamcs_combined.csv'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    print("Loading model and data...")
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # Sample data for speed (SHAP is slow)
    df_sample = df.sample(n=1000, random_state=42).dropna(subset=['ESI'])
    
    # 2. Prepare Data
    numeric_features = ['Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat', 'PainScale']
    categorical_features = ['ArrivalMode']
    text_feature = 'Chief_complain'
    
    X = df_sample[numeric_features + categorical_features + [text_feature]]
    
    # 3. Transform Data
    print("Transforming data...")
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X)
    
    # Get Feature Names
    try:
        feature_names = get_feature_names(preprocessor)
    except Exception as e:
        print(f"Could not extract feature names automatically: {e}")
        # Fallback: Generate generic names
        feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]

    # 4. Extract LightGBM Model
    # The classifier is the last step
    stacking_clf = pipeline.named_steps['classifier']
    # Access the first estimator (LGBM)
    # Note: estimators_ is a list of fitted estimators
    lgbm_model = stacking_clf.estimators_[0]
    print(f"Explaining model: {type(lgbm_model).__name__}")

    # 5. Calculate SHAP Values
    print("Calculating SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(lgbm_model)
    
    # Convert sparse matrix to dense if necessary (LGBM handles sparse, but SHAP might prefer dense or specific format)
    if hasattr(X_transformed, "toarray"):
        X_transformed_dense = X_transformed.toarray()
    else:
        X_transformed_dense = X_transformed

    shap_values = explainer.shap_values(X_transformed_dense)

    # 6. Plotting
    os.makedirs('output/plots', exist_ok=True)
    
    # Summary Plot (Bar) - Global Importance
    plt.figure()
    # For multi-class, shap_values is a list of arrays (one for each class).
    # We'll plot the summary for Class 0 (ESI 1 - Critical) or aggregate.
    # Let's plot for all classes combined if possible, or just ESI 3 (most common/tricky).
    # Or better: Summary plot of Class 2 (ESI 3) which is the "grey area".
    
    # Check if shap_values is list (multiclass) or array (binary)
    if isinstance(shap_values, list):
        print(f"Model has {len(shap_values)} classes. Plotting summary for ESI 3 (Index 2)...")
        # Index 2 corresponds to ESI 3 (since 0=ESI 1, 1=ESI 2, 2=ESI 3...)
        target_class_idx = 2 
        shap_vals_target = shap_values[target_class_idx]
    else:
        shap_vals_target = shap_values

    shap.summary_plot(shap_vals_target, X_transformed_dense, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot (ESI 3 Prediction)")
    plt.tight_layout()
    plt.savefig('output/plots/shap_summary_esi3.png')
    plt.close()
    
    # Bar plot of feature importance (mean abs shap value)
    plt.figure()
    shap.summary_plot(shap_vals_target, X_transformed_dense, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig('output/plots/shap_importance_bar.png')
    plt.close()

    print("SHAP plots generated in output/plots/")

if __name__ == "__main__":
    main()
