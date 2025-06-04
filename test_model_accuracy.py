#!/usr/bin/env python3
"""
Test the accuracy of the CATastrophe model using test datasets.
"""

import json
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from catastrophe.model.autoencoder import Autoencoder

def main():
    # Load test data
    print("Loading test data...")
    with open('tests/test_data/safe_c_commits.json', 'r') as f:
        safe_commits = json.load(f)
    
    with open('tests/test_data/vulnerable_c_commits.json', 'r') as f:
        vulnerable_commits = json.load(f)
    
    print(f"Loaded {len(safe_commits)} safe commits")
    print(f"Loaded {len(vulnerable_commits)} vulnerable commits")
    
    # Download model from Hugging Face
    model_repo = "ewhk9887/CATastrophe"
    print(f"\nDownloading model from {model_repo}...")
    
    try:
        model_path = hf_hub_download(repo_id=model_repo, filename="catastrophe_model.pth")
        vectorizer_path = hf_hub_download(repo_id=model_repo, filename="vectorizer.pkl")
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return
    
    # Load vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    feature_dim = vectorizer.get_feature_names_out().shape[0]
    print(f"Feature dimension: {feature_dim}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Autoencoder(feature_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Test function
    def predict_vulnerability(message, func, threshold=0.5):
        text = f"{message} {func}"
        features = vectorizer.transform([text]).toarray()
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            reconstruction = model(features_tensor)
        
        mse = torch.nn.functional.mse_loss(reconstruction, features_tensor, reduction='none')
        reconstruction_error = mse.mean(dim=1).item()
        
        return reconstruction_error > threshold, reconstruction_error
    
    # Evaluate
    print("\nEvaluating model...")
    y_true = []
    y_pred = []
    y_scores = []
    
    # Test safe commits
    for commit in safe_commits:
        is_vuln, score = predict_vulnerability(commit['message'], commit['func'])
        y_true.append(0)
        y_pred.append(1 if is_vuln else 0)
        y_scores.append(score)
    
    # Test vulnerable commits
    for commit in vulnerable_commits:
        is_vuln, score = predict_vulnerability(commit['message'], commit['func'])
        y_true.append(1)
        y_pred.append(1 if is_vuln else 0)
        y_scores.append(score)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"RESULTS (Default threshold=0.5)")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Safe', 'Vulnerable']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Safe  Vulnerable")
    print(f"Actual Safe     {cm[0][0]:<5} {cm[0][1]:<5}")
    print(f"Actual Vuln     {cm[1][0]:<5} {cm[1][1]:<5}")
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        print(f"\nROC AUC Score: {roc_auc:.3f}")
    except:
        print("\nCould not calculate ROC AUC")
    
    # Find optimal threshold
    print(f"\n{'='*50}")
    print("Finding optimal threshold...")
    print(f"{'='*50}")
    
    best_threshold = None
    best_f1 = 0
    
    for threshold in np.arange(0.1, 1.0, 0.05):
        y_pred_temp = [1 if score > threshold else 0 for score in y_scores]
        report = classification_report(y_true, y_pred_temp, output_dict=True)
        f1 = report['weighted avg']['f1-score']
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Best F1 Score: {best_f1:.2%}")
    
    # Re-evaluate with optimal threshold
    y_pred_optimal = [1 if score > best_threshold else 0 for score in y_scores]
    accuracy_optimal = accuracy_score(y_true, y_pred_optimal)
    print(f"\nAccuracy with optimal threshold: {accuracy_optimal:.2%}")
    print("\nClassification Report with optimal threshold:")
    print(classification_report(y_true, y_pred_optimal, target_names=['Safe', 'Vulnerable']))
    
    # Example predictions
    print(f"\n{'='*50}")
    print("Example Predictions")
    print(f"{'='*50}")
    
    print("\nVulnerable Examples:")
    for i, commit in enumerate(vulnerable_commits[:3]):
        is_vuln, score = predict_vulnerability(commit['message'], commit['func'], threshold=best_threshold)
        print(f"\n{i+1}. Message: {commit['message']}")
        print(f"   Score: {score:.4f}")
        print(f"   Predicted: {'Vulnerable' if is_vuln else 'Safe'}")
        print(f"   Correct: {'✓' if is_vuln else '✗'}")
    
    print("\n\nSafe Examples:")
    for i, commit in enumerate(safe_commits[:3]):
        is_vuln, score = predict_vulnerability(commit['message'], commit['func'], threshold=best_threshold)
        print(f"\n{i+1}. Message: {commit['message']}")
        print(f"   Score: {score:.4f}")
        print(f"   Predicted: {'Vulnerable' if is_vuln else 'Safe'}")
        print(f"   Correct: {'✓' if not is_vuln else '✗'}")
    
    # Score distribution analysis
    safe_scores = y_scores[:len(safe_commits)]
    vuln_scores = y_scores[len(safe_commits):]
    
    print(f"\n{'='*50}")
    print("Score Distribution Analysis")
    print(f"{'='*50}")
    print(f"Safe commits - Mean: {np.mean(safe_scores):.4f}, Std: {np.std(safe_scores):.4f}")
    print(f"Vulnerable commits - Mean: {np.mean(vuln_scores):.4f}, Std: {np.std(vuln_scores):.4f}")

if __name__ == "__main__":
    main()