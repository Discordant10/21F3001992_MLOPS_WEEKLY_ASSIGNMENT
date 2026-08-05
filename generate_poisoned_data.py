import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def generate_datasets(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load clean dataset
    iris = load_iris(as_frame=True)
    df_clean = iris.frame
    df_clean.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    
    # Save baseline 0% clean dataset
    clean_path = os.path.join(output_dir, "iris_clean.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved baseline clean dataset to {clean_path}")

    # Determine feature bounds and target classes for random noise injection
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    min_vals = df_clean[feature_cols].min().to_dict()
    max_vals = df_clean[feature_cols].max().to_dict()
    unique_labels = df_clean['target'].unique()

    poison_levels = [0.05, 0.10, 0.50]
    np.random.seed(42)

    for pct in poison_levels:
        df_poisoned = df_clean.copy()
        n_samples = len(df_poisoned)
        n_poison = int(n_samples * pct)

        # Select random indices to corrupt
        poison_indices = np.random.choice(n_samples, size=n_poison, replace=False)

        for idx in poison_indices:
            # Replace all 4 feature values with uniform random values within original feature range
            for col in feature_cols:
                df_poisoned.loc[idx, col] = np.round(
                    np.random.uniform(min_vals[col], max_vals[col]), 2
                )
            # Assign random class label
            df_poisoned.loc[idx, 'target'] = np.random.choice(unique_labels)

        out_file = os.path.join(output_dir, f"iris_poisoned_{int(pct*100)}.csv")
        df_poisoned.to_csv(out_file, index=False)
        print(f"Saved {int(pct*100)}% poisoned dataset ({n_poison} rows corrupted) to {out_file}")

if __name__ == "__main__":
    generate_datasets()