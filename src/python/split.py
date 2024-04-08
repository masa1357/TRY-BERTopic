"""
Make a split of the data into train, valid and test sets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("../data/newdf.csv")

    # Split the data
    train, temp = train_test_split(data, test_size=0.2, random_state=42)
    valid, test = train_test_split(temp, test_size=0.5, random_state=42)

    print(f"Train shape: {train.shape}\nValid shape: {valid.shape}\nTest shape: {test.shape}")
    
    