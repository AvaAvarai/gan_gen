import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

# Load the original and synthetic Iris datasets
original_data = pd.read_csv('original_iris.csv')
synthetic_data = pd.read_csv('synthetic_iris_5000.csv')

# Map the target column to string values for proper visualization
class_mapping = {'Setosa': 'Setosa', 'Versicolor': 'Versicolor', 'Virginica': 'Virginica'}
original_data['target'] = original_data['target'].map(class_mapping)
synthetic_data['target'] = synthetic_data['target'].map(class_mapping)

# Normalize the feature columns using MinMaxScaler
scaler = MinMaxScaler()

# Separate features and target
original_features = original_data.drop('target', axis=1)
synthetic_features = synthetic_data.drop('target', axis=1)

# Fit and transform the scaler on both datasets
original_features_normalized = pd.DataFrame(scaler.fit_transform(original_features), columns=original_features.columns)
synthetic_features_normalized = pd.DataFrame(scaler.fit_transform(synthetic_features), columns=synthetic_features.columns)

# Add the target column back to the normalized data
original_data_normalized = original_features_normalized.copy()
original_data_normalized['target'] = original_data['target']

synthetic_data_normalized = synthetic_features_normalized.copy()
synthetic_data_normalized['target'] = synthetic_data['target']

# Check for NaN values in the target column
print("Original data NaN values:\n", original_data.isna().sum())
print("Synthetic data NaN values:\n", synthetic_data.isna().sum())

# Ensure there are no NaN values in the target column
assert not original_data['target'].isna().any(), "NaN values found in original data 'target' column"
assert not synthetic_data['target'].isna().any(), "NaN values found in synthetic data 'target' column"

# Define colors for each class
colors = ['blue', 'green', 'red']

# Plot parallel coordinates for the original data
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
parallel_coordinates(original_data_normalized, 'target', color=colors, alpha=0.8, linewidth=1)
plt.title('Parallel Coordinates Plot of Original Iris Data')
plt.xlabel('Attributes')
plt.ylabel('Value')
plt.legend(handles=[plt.Line2D([], [], color='blue', label='Setosa'),
                    plt.Line2D([], [], color='green', label='Versicolor'),
                    plt.Line2D([], [], color='red', label='Virginica')])
plt.grid(True)

# Plot parallel coordinates for the synthetic data
plt.subplot(1, 2, 2)
parallel_coordinates(synthetic_data_normalized, 'target', color=colors, alpha=0.8, linewidth=1)
plt.title('Parallel Coordinates Plot of Synthetic Iris Data')
plt.xlabel('Attributes')
plt.ylabel('Value')
plt.legend(handles=[plt.Line2D([], [], color='blue', label='Setosa'),
                    plt.Line2D([], [], color='green', label='Versicolor'),
                    plt.Line2D([], [], color='red', label='Virginica')])
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('parallel_coordinates_normalized_iris_5000.png')
plt.show()
