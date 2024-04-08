import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_sprit(random_state=42):
    # Load the data
    df = pd.read_csv("../data/newdf.csv")
    # Split the data(train, valid, test)
    train, temp = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df['label'])
    valid, test = train_test_split(temp, test_size=0.5, random_state=random_state, stratify=temp['label'])
    print("Train shape: ", train.shape)
    print("Valid shape: ", valid.shape)
    print("Test shape: ", test.shape)
    return train, valid, test
     
if __name__=="__main__":
    train, valid, test = train_test_sprit()