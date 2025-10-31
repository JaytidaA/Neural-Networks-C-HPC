# clean.py: Cleanup the electrcity_cost_dataset.csv
# https://www.kaggle.com/datasets/shalmamuji/electricity-cost-prediction-dataset
# Written by Jojo (Aaditya Joil): 231080033

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('./datasets/heart_cleveland_upload.csv')

    # Normalise the input fields numeric
    for col in df.select_dtypes(include = "number").columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

    df_train = X_train
    df_train["structure type"] = y_train
    df_test = X_test
    df_test["structure type"] = y_test

    # Save the datasets accordingly
    df_train.to_csv(f"./datasets/heart_disease_train.csv", index = False)
    df_test.to_csv(f"./datasets/heart_disease_test.csv", index = False)

if __name__ == "__main__":
    main()
