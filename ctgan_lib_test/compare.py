import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import hsv_to_rgb

# Load the original Iris dataset
original_data = pd.read_csv('original_iris.csv')

# Get the unique class names and sort them alphabetically
class_names = sorted(original_data['target'].unique())

# Dynamically generate a color palette by subdividing the HSV space
hsv_colors = [(i / len(class_names), 1, 1) for i in range(len(class_names))]
rgb_colors = [hsv_to_rgb(hsv) for hsv in hsv_colors]
class_palette = {class_name: rgb_colors[i] for i, class_name in enumerate(class_names)}

# Function to add a single legend
def add_legend(fig, class_order, class_palette):
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_palette[class_name], markersize=10) for class_name in class_order]
    labels = class_order
    fig.legend(handles, labels, loc='center right', title='Classes')

# Normalize the feature columns
scaler = StandardScaler()
original_data[original_data.columns[:-1]] = scaler.fit_transform(original_data[original_data.columns[:-1]])

# Create and save pairplot for the original Iris data using the consistent palette
pairplot_original = sns.pairplot(original_data, hue='target', palette=class_palette)
pairplot_original.fig.suptitle('Original Iris Data', y=1.02)

# Add a single legend to the original data plot
add_legend(pairplot_original.fig, class_names, class_palette)

pairplot_original.fig.savefig('pairplot_original_iris.png')
plt.close(pairplot_original.fig)  # Close the figure to save memory

# Define the list of epochs
epochs_list = range(1000, 10001, 1000)

# Directory containing the synthetic datasets
results_dir = 'results2'

# Create pair plots for each synthetic dataset
for epochs in epochs_list:
    synthetic_data_path = os.path.join(results_dir, f'synthetic_iris_complete_{epochs}.csv')
    
    # Load the synthetic dataset
    synthetic_data = pd.read_csv(synthetic_data_path)
    
    # Normalize the feature columns
    synthetic_data[synthetic_data.columns[:-1]] = scaler.transform(synthetic_data[synthetic_data.columns[:-1]])
    
    # Create and save pairplot for the synthetic Iris data using the consistent palette
    pairplot_synthetic = sns.pairplot(synthetic_data, hue='target', palette=class_palette)
    pairplot_synthetic.fig.suptitle(f'Synthetic Data of Iris at {epochs} Epochs', y=1.02)
    
    # Add a single legend to the synthetic data plot
    add_legend(pairplot_synthetic.fig, class_names, class_palette)

    pairplot_synthetic.fig.savefig(f'pairplot_synthetic_iris_{epochs}.png')
    plt.close(pairplot_synthetic.fig)  # Close the figure to save memory

print("All pair plots generated and saved.")
