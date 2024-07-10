import os
from ctgan import CTGAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Map the target column to class names
class_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
data['target'] = data['target'].map(class_mapping)

# Normalize the feature columns
scaler = StandardScaler()
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])

# Save the original Iris data to a CSV file
data.to_csv('original_iris.csv', index=False)

# Create the results directory if it does not exist
os.makedirs('results2', exist_ok=True)

# List of epochs to run the model with
epochs_list = range(1000, 10001, 1000)

for epochs in epochs_list:
    # Define the CTGAN model with the specified number of epochs
    ctgan = CTGAN(epochs=epochs, batch_size=500, generator_lr=2e-4, discriminator_lr=2e-4)

    # Fit the model to the data
    ctgan.fit(data, discrete_columns=['target'])

    # Create synthetic data
    synthetic_data = ctgan.sample(1000)  # Generate more samples than needed

    # Save the complete synthetic dataset to a CSV file
    synthetic_data.to_csv(f'results2/synthetic_iris_complete_{epochs}.csv', index=False)

    # Ensure we have 50 samples for each class
    synthetic_samples = pd.concat([
        synthetic_data[synthetic_data['target'] == class_name].sample(50, replace=True)
        for class_name in class_mapping.values()
    ])

    # Shuffle the samples to mix the classes
    synthetic_samples = synthetic_samples.sample(frac=1).reset_index(drop=True)

    # Inverse transform the features to original scale
    synthetic_samples[synthetic_samples.columns[:-1]] = scaler.inverse_transform(synthetic_samples[synthetic_samples.columns[:-1]])

    # Save the balanced synthetic data to a CSV file
    synthetic_samples.to_csv(f'results2/synthetic_iris_balanced_{epochs}.csv', index=False)

    print(f"Complete Synthetic Iris data saved to 'results2/synthetic_iris_complete_{epochs}.csv'")
    print(f"Balanced Synthetic Iris data saved to 'results2/synthetic_iris_balanced_{epochs}.csv'")
    print(synthetic_samples.head())
    print(synthetic_samples['target'].value_counts())

print("Original Iris data saved to 'original_iris.csv'")
print("Synthetic Iris datasets saved to 'results2' directory")
