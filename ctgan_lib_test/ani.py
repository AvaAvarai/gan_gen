import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Define a consistent color palette for the classes
class_palette = {'Setosa': 'blue', 'Versicolor': 'orange', 'Virginica': 'green'}

# Load the original Iris dataset
original_data = pd.read_csv('original_iris.csv')

# Create and save pairplot for the original Iris data using the consistent palette
pairplot_original = sns.pairplot(original_data, hue='target', palette=class_palette)
pairplot_original.fig.suptitle('Original Iris Data', y=1.02)
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
    
    # Create and save pairplot for the synthetic Iris data using the consistent palette
    pairplot_synthetic = sns.pairplot(synthetic_data, hue='target', palette=class_palette)
    pairplot_synthetic.fig.suptitle(f'Synthetic Iris Data (epochs={epochs})', y=1.02)
    pairplot_synthetic.fig.savefig(f'pairplot_synthetic_iris_{epochs}.png')
    plt.close(pairplot_synthetic.fig)  # Close the figure to save memory

print("All pair plots generated and saved.")

# Create an animation from the saved pair plot images
fig, ax = plt.subplots(figsize=(10, 6))

# List of image files to animate
image_files = [f'pairplot_synthetic_iris_{epochs}.png' for epochs in epochs_list]

# Function to update the frame
def update(frame):
    img = plt.imread(image_files[frame])
    ax.imshow(img)
    ax.set_axis_off()

# Create the animation
ani = FuncAnimation(fig, update, frames=len(image_files), repeat=True, interval=1000)

# Save the animation as a GIF file
ani.save('synthetic_iris_animation.gif', writer='imagemagick')

print("Animation created and saved as 'synthetic_iris_animation.gif'.")
plt.show()
