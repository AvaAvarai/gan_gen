import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load original and synthetic data
original_df = pd.read_csv('data/fisher_iris.csv')
synthetic_df = pd.read_csv('synthetic_iris_data.csv')

# Ensure categorical column 'class' in the synthetic data is in the same form as in original
synthetic_df['class'] = synthetic_df['class'].astype(str)
original_df['class'] = original_df['class'].astype(str)

# Scatterplot Matrix for Original Data
fig_orig_sp = px.scatter_matrix(original_df, dimensions=original_df.columns[:-1], color='class',
                                title="Scatterplot Matrix - Original Data")
fig_orig_sp.show()

# Scatterplot Matrix for Synthetic Data
fig_synth_sp = px.scatter_matrix(synthetic_df, dimensions=synthetic_df.columns[:-1], color='class',
                                 title="Scatterplot Matrix - Synthetic Data")
fig_synth_sp.show()

# Parallel Coordinates Plot for Original Data
fig_orig_pc = px.parallel_coordinates(original_df, color='class',
                                      labels={col: col.replace('.', ' ') for col in original_df.columns},
                                      title="Parallel Coordinates - Original Data")
fig_orig_pc.show()

# Parallel Coordinates Plot for Synthetic Data
fig_synth_pc = px.parallel_coordinates(synthetic_df, color='class',
                                       labels={col: col.replace('.', ' ') for col in synthetic_df.columns},
                                       title="Parallel Coordinates - Synthetic Data")
fig_synth_pc.show()
