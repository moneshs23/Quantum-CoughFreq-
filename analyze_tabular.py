import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dataset(csv_path):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("\n--- Dataset Distribution ---")
    print(df['status'].value_counts())
    
    # Calculate some key features for quantum mapping
    # MFCCs are columns 0-12 usually or named MFCC_1..13
    mfcc_cols = [col for col in df.columns if 'MFCC' in col]
    
    print("\n--- Feature Statistics (Top 5 MFCCs) ---")
    stats = df[mfcc_cols[:5]].describe().loc[['mean', 'std']]
    print(stats)
    
    # Simulate Quantum State Injection (based on project patterns)
    print("\n--- Simulated Quantum Analysis Output ---")
    results = []
    for label in df['status'].unique():
        subset = df[df['status'] == label]
        avg_spectral_centroid = subset['SpectralCentroid'].mean()
        avg_zcr = subset['ZeroCrossingRate'].mean() if 'ZeroCrossingRate' in subset.columns else 0
        
        # Simple heuristic mapping for demo output
        prob_tb = 0.0
        if label == 'COVID-19':
            prob_tb = 0.45 + (avg_spectral_centroid / 8000.0) * 0.2
        elif label == 'symptomatic':
            prob_tb = 0.25 + (avg_spectral_centroid / 8000.0) * 0.1
        else:
            prob_tb = 0.05 + (avg_spectral_centroid / 8000.0) * 0.05
            
        results.append({
            "Label": label,
            "Count": len(subset),
            "Avg Spectral Centroid": round(avg_spectral_centroid, 2),
            "Quantum TB Prob (Sim)": round(prob_tb, 4)
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    return res_df

if __name__ == "__main__":
    path = "/Users/moni/Desktop/Qml/tabular_form/tabular_form/extracted_features_coughvid_v3.csv"
    analyze_dataset(path)
