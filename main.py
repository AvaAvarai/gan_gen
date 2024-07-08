# main.py
import pandas as pd
from src.train import train_ctgan
from src.generate import generate_synthetic_data

def main():
    # Load the Iris dataset from the provided CSV file
    df = pd.read_csv('data/fisher_iris.csv')
    
    # Train the CTGAN model
    generator, scaler, encoders, processed_columns = train_ctgan(df)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(generator, scaler, encoders, processed_columns, df, num_samples_per_class=50)
    
    # Verify the first few rows of synthetic data
    print("First few rows of the synthetic data:")
    print(synthetic_data.head())

    # Save synthetic data
    synthetic_data.to_csv('synthetic_iris_data.csv', index=False)
    print("Synthetic Iris data saved to synthetic_iris_data.csv")

if __name__ == "__main__":
    main()
