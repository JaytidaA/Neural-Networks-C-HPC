# heart_disease.py: Comparision of MLPClassifier with the 
# Written by Jojo (Aaditya Joil): 231080033

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('./datasets/heart_cleveland_upload.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Scale inputs
    for col in X.columns:
        mean = X[col].mean()
        std  = X[col].std()
        X[col] = (X[col] - mean) / std

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.values
    X_test = X_test.values

    print(f"--- Testing with sklearn neural network ---")
    mlp_model = MLPClassifier(
        hidden_layer_sizes = (16, 16,),
        activation = 'relu',
        solver = 'adam',
        max_iter = 10000,
        random_state = 42,
    )
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)

    print(f"Accuracy score for neural network: {accuracy_score(y_test, y_pred)}")

if __name__ == '__main__':
    main()