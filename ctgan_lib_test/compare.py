import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the original and synthetic Iris datasets
original_data = pd.read_csv('original_iris.csv')
synthetic_data = pd.read_csv('synthetic_iris_5000.csv')

# Create pairplot for the original Iris data and save it to a file
pairplot_original = sns.pairplot(original_data, hue='target')
pairplot_original.fig.suptitle('Original Iris Data', y=1.02)
pairplot_original.fig.savefig('pairplot_original_iris.png')

# Create pairplot for the synthetic Iris data and save it to a file
pairplot_synthetic = sns.pairplot(synthetic_data, hue='target')
pairplot_synthetic.fig.suptitle('Synthetic Iris Data', y=1.02)
pairplot_synthetic.fig.savefig('pairplot_synthetic_iris_5000.png')

# close the pairplot figures
plt.close('all')

# Load the saved pairplot images
original_image = plt.imread('pairplot_original_iris.png')
synthetic_image = plt.imread('pairplot_synthetic_iris_5000.png')

# Create a new figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Display the original pairplot image
axes[0].imshow(original_image)
axes[0].axis('off')  # Hide the axes
axes[0].set_title('Original Iris Data')

# Display the synthetic pairplot image
axes[1].imshow(synthetic_image)
axes[1].axis('off')  # Hide the axes
axes[1].set_title('Synthetic Iris Data')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
