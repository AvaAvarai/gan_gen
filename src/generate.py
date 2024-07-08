# src/generate.py
import torch
import pandas as pd

def generate_synthetic_data(generator, scaler, encoders, processed_columns, original_df, num_samples_per_class=50, latent_dim=100):
    synthetic_data_list = []
    
    for class_value in encoders['class'].classes_:
        # Generate synthetic data for each class
        class_index = encoders['class'].transform([class_value])[0]
        z = torch.randn(num_samples_per_class, latent_dim)
        synthetic_data = generator(z).detach().numpy()
        synthetic_data_df = pd.DataFrame(synthetic_data, columns=processed_columns)
        
        # Set the class column to the current class
        synthetic_data_df['class'] = class_index
        
        # Inverse transform numerical columns
        numerical_columns = original_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            synthetic_data_df[numerical_columns] = scaler.inverse_transform(synthetic_data_df[numerical_columns])
        
        # Inverse transform categorical columns
        for col in ['class']:
            if col in synthetic_data_df.columns:
                encoder = encoders[col]
                synthetic_data_df[col] = encoder.inverse_transform(synthetic_data_df[col].round().astype(int))
        
        synthetic_data_list.append(synthetic_data_df)
    
    # Concatenate all synthetic data into one DataFrame
    synthetic_data_df = pd.concat(synthetic_data_list, ignore_index=True)
    
    return synthetic_data_df
