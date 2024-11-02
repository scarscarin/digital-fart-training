import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import os
import pynvml
import time

# List of audio file names
audio_files = [f'Recording_{i}.m4a' for i in range(1, 7)]

# Directory containing the audio files
audio_dir = 'audio'

# Load audio files and store them in a list
audio_data_list = []
for file in audio_files:
    file_path = os.path.join(audio_dir, file)
    # Load audio using librosa
    waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
    audio_data_list.append(waveform)

# Find the minimum length among all audio files
min_length = min([len(waveform) for waveform in audio_data_list])

# Truncate all audio files to the minimum length and stack them
audio_data = np.stack([waveform[:min_length] for waveform in audio_data_list])

# Convert audio data to PyTorch tensor
audio_tensor = torch.from_numpy(audio_data).float()

# Normalize the data to [-1, 1]
audio_min = audio_tensor.min().item()
audio_max = audio_tensor.max().item()
audio_tensor = (audio_tensor - audio_min) / (audio_max - audio_min)
audio_tensor = audio_tensor * 2 - 1  # Scale to [-1, 1]

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Flatten the audio data for training
input_dim = audio_tensor.shape[1]
audio_tensor_flat = audio_tensor

# Create a DataLoader
dataset = torch.utils.data.TensorDataset(audio_tensor_flat)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_energy = 0  # Energy in joules
    epoch_start_time = time.time()
    for data in dataloader:
        input_data = data[0]
        # Start power measurement
        power_usage_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        # Forward pass
        output = model(input_data)
        loss = criterion(output, input_data)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # End power measurement
        power_usage_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        power_usage = power_usage_end - power_usage_start  # Energy in millijoules
        epoch_energy += power_usage
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Duration in seconds
    # Convert energy from millijoules to joules
    epoch_energy_joules = epoch_energy / 1000
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Energy Consumption: {epoch_energy_joules:.2f} J')

# Shutdown NVML
pynvml.nvmlShutdown()

# Use the encoder to get latent representations
with torch.no_grad():
    latent_vectors = model.encoder(audio_tensor_flat)

# Generate a new latent vector by averaging and adding noise
new_latent = torch.mean(latent_vectors, dim=0, keepdim=True)
noise = torch.randn_like(new_latent) * 0.1
new_latent += noise

# Decode the new latent vector to get the new audio
new_audio = model.decoder(new_latent)

# Rescale to original amplitude
new_audio = new_audio.detach().numpy()
new_audio = (new_audio + 1) / 2  # Scale back to [0, 1]
new_audio = new_audio * (audio_max - audio_min) + audio_min

import soundfile as sf

# Convert the numpy array to a 1D array
new_audio = new_audio.squeeze()

# Save the audio file
output_file = 'new_audio.wav'
sf.write(output_file, new_audio, samplerate=sample_rate)
