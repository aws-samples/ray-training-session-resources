# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import pandas as pd
import numpy as np

def generate_sample_data(n_samples=100000):
    np.random.seed(42)
    
    # Generate random features (764 features as per your model)
    X = np.random.randn(n_samples, 764)
    
    # Generate target (10 classes as per your model's output)
    y = np.random.randint(0, 10, n_samples)
    
    # Create DataFrame
    columns = [f'feature_{i}' for i in range(764)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    
    # Save to CSV
    df.to_csv('sample_data.csv', index=False)
    return df

generate_sample_data()