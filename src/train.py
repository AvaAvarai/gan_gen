# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.data_preprocessor import preprocess_data
from src.models import Generator, Discriminator

def train_ctgan(df, latent_dim=100, hidden_dim=128, batch_size=32, num_epochs=200, learning_rate=0.0002):
    # Preprocess the data
    processed_df, scaler, encoders = preprocess_data(df)
    
    input_dim = processed_df.shape[1]
    
    # Initialize models
    generator = Generator(latent_dim, hidden_dim, input_dim)
    discriminator = Discriminator(input_dim, hidden_dim)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(processed_df), batch_size):
            # Get real and fake data batches
            real_data = torch.tensor(processed_df.iloc[i:i+batch_size].values, dtype=torch.float32)
            batch_size_real = real_data.size(0)
            labels_real = torch.ones(batch_size_real, 1)
            labels_fake = torch.zeros(batch_size_real, 1)
    
            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(batch_size_real, latent_dim)
            fake_data = generator(z)
            output_real = discriminator(real_data)
            output_fake = discriminator(fake_data.detach())
            loss_D = criterion(output_real, labels_real) + criterion(output_fake, labels_fake)
            loss_D.backward()
            optimizer_D.step()
    
            # Train Generator
            optimizer_G.zero_grad()
            output_fake = discriminator(fake_data)
            loss_G = criterion(output_fake, labels_real)  # Trick discriminator
            loss_G.backward()
            optimizer_G.step()
    
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_D.item()}, Loss G: {loss_G.item()}')
    
    return generator, scaler, encoders, processed_df.columns
