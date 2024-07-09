# src/data_preprocessor.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df):
    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'string']).columns

    # Normalize numerical columns
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Encode categorical columns
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    return df, scaler, encoders

if __name__ == "__main__":
    df = pd.read_csv('data/fisher_iris.csv')
    processed_df, scaler, encoders = preprocess_data(df)
    print(processed_df.head())
