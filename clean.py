# clean.py: Cleanup the electrcity_cost_dataset.csv
# https://www.kaggle.com/datasets/shalmamuji/electricity-cost-prediction-dataset
# Written by Jojo (Aaditya Joil): 231080033

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('./datasets/electricity_cost_dataset.csv')

    # Normalise the input fields numeric
    for col in df.select_dtypes(include = "number").columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std

    # Get the "structure type" column and Label Encode (0 starting) it and make it the last column
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["structure type"] = le.fit_transform(df["structure type"])

    structure_col = df.pop("structure type")
    df["structure type"] = structure_col
 

    # Save the datasets accordingly
    df.to_csv(f"./datasets/electricity_cost_clean.csv", index = False)

if __name__ == "__main__":
    main()
