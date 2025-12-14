import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import os

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_score, n_classes, classes, output_dir):
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_score, n_classes, classes, output_dir):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {classes[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_class_distribution(y, classes, output_dir):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('ESI Level')
    plt.ylabel('Count')
    plt.xticks(ticks=range(len(classes)), labels=classes)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()

def apply_safety_net(X_test, y_pred, classes):
    print("Applying Safety Net Logic...")
    y_pred_safe = y_pred.copy()
    
    # Get indices
    try:
        esi_2_idx = np.where(classes == 2.0)[0][0]
        esi_1_idx = np.where(classes == 1.0)[0][0]
    except IndexError:
        print("Error: ESI classes 1.0 or 2.0 not found.")
        return y_pred

    # Counters
    upgraded_count = 0
    
    # Iterate
    # Reset index of X_test to match y_pred array
    X_test_reset = X_test.reset_index(drop=True)
    
    for i in range(len(y_pred)):
        current_pred = y_pred_safe[i]
        row = X_test_reset.iloc[i]
        
        # Don't downgrade ESI 1 (Index 0)
        if current_pred == esi_1_idx:
            continue
            
        # Safety Rules (Trigger ESI 2)
        is_critical = False
        
        # Vitals (Check for validity first to avoid NaN issues, though comparison usually handles it)
        if row['O2Sat'] < 90: is_critical = True
        if row['Pulse'] > 110: is_critical = True # Lowered threshold slightly
        if row['SBP'] < 90: is_critical = True
        if row['Resp'] > 24: is_critical = True
        if row['Temp'] > 103 or row['Temp'] < 95: is_critical = True
        
        # Pain
        if row['PainScale'] >= 7: is_critical = True
        
        # Arrival Mode (EMS = 1.0)
        if row['ArrivalMode'] == 1.0: is_critical = True

        if is_critical:
            # Only upgrade if current prediction is lower acuity (Higher Index)
            # ESI 2 is Index 1. ESI 3,4,5 are Indices 2,3,4.
            if current_pred > esi_2_idx:
                y_pred_safe[i] = esi_2_idx
                upgraded_count += 1
                
    print(f"Safety Net upgraded {upgraded_count} patients to ESI 2.")
    return y_pred_safe

def main():
    model_path = 'data/nhamcs_model.joblib'
    data_path = 'data/nhamcs_combined.csv'
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading model from {model_path}...")
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['ESI'])
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['ESI'])
    classes = le.classes_ # Original ESI labels (1.0, 2.0, etc)
    n_classes = len(classes)
    
    # Updated Features
    numeric_features = ['Age', 'Temp', 'Pulse', 'Resp', 'SBP', 'DBP', 'O2Sat', 'PainScale']
    categorical_features = ['ArrivalMode']
    text_feature = 'Chief_complain'
    
    X = df[numeric_features + categorical_features + [text_feature]]
    
    # Plot Class Distribution
    plot_class_distribution(y, classes, output_dir)
    
    print("\nRunning Train/Test Split Evaluation (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Clone pipeline to ensure fresh training
    eval_pipeline = clone(pipeline)
    
    print("Training on 80% of data...")
    eval_pipeline.fit(X_train, y_train)
    
    print("Predicting on 20% test set...")
    y_test_pred = eval_pipeline.predict(X_test)
    y_test_prob = eval_pipeline.predict_proba(X_test)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Set Accuracy: {test_acc:.4f}")
    
    # Save Classification Report
    report = classification_report(y_test, y_test_pred, target_names=[str(c) for c in classes])
    print("\nTest Set Classification Report:")
    print(report)
    
    # --- Apply Safety Net ---
    y_test_pred_safe = apply_safety_net(X_test, y_test_pred, classes)
    
    test_acc_safe = accuracy_score(y_test, y_test_pred_safe)
    print(f"\nSafety Net Test Accuracy: {test_acc_safe:.4f}")
    
    report_safe = classification_report(y_test, y_test_pred_safe, target_names=[str(c) for c in classes])
    print("\nSafety Net Classification Report:")
    print(report_safe)
    # ------------------------
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Set Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\n------------------------------------------------\n")
        f.write(f"Safety Net Test Accuracy: {test_acc_safe:.4f}\n\n")
        f.write("Safety Net Classification Report:\n")
        f.write(report_safe)
    
    # Generate Plots (Using Safety Net Predictions for Confusion Matrix)
    print("Generating plots (using Safety Net predictions)...")
    plot_confusion_matrix(y_test, y_test_pred_safe, classes, output_dir)
    # ROC/PR curves require probabilities, which we can't easily adjust with rules, 
    # so we keep the original model's curves or we'd need to hack the probs.
    # For now, we just plot the original curves to show model capability.
    plot_roc_curve(y_test, y_test_prob, n_classes, classes, output_dir)
    plot_precision_recall_curve(y_test, y_test_prob, n_classes, classes, output_dir)
    
    print(f"All outputs saved to {output_dir}/")

if __name__ == "__main__":
    main()
