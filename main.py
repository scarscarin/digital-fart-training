import os
import time
import requests
import dropbox
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
# import pynvml
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your Dropbox app credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')

# Dropbox token endpoint and authorization URL
TOKEN_URL = "https://api.dropbox.com/oauth2/token"
AUTH_URL = f"https://www.dropbox.com/oauth2/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&token_access_type=offline"

# Step 1: Direct the user to authorize your app
print("Go to this URL and authorize the app:", AUTH_URL)

# Step 2: After user authorization, they will be redirected to your REDIRECT_URI with a 'code' in the URL
AUTHORIZATION_CODE = input("Enter the authorization code: ")

# Step 3: Exchange the authorization code for access and refresh tokens
data = {
    'code': AUTHORIZATION_CODE,
    'grant_type': 'authorization_code',
    'redirect_uri': REDIRECT_URI,
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET
}

response = requests.post(TOKEN_URL, data=data)
response_data = response.json()

# Extract tokens
access_token = response_data.get('access_token')
refresh_token = response_data.get('refresh_token')

print("Access Token:", access_token)
print("Refresh Token:", refresh_token)

# Initialize Dropbox client with the access token
dbx = dropbox.Dropbox(access_token)

# Folder in your Dropbox containing the audio files
dropbox_folder_path = '/audio'

# List all files in the Dropbox folder
file_entries = dbx.files_list_folder(dropbox_folder_path).entries

# Download and process audio files
audio_data_list = []
for entry in file_entries:
    if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.mp3'):
        # Construct the full path for Dropbox
        dropbox_path = f"{dropbox_folder_path}/{entry.name}"
        # Download file from Dropbox
        _, res = dbx.files_download(dropbox_path)
        
        # Save the audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(res.content)
            temp_audio_file_path = temp_audio_file.name
        
        # Load audio using librosa
        waveform, sample_rate = librosa.load(temp_audio_file_path, sr=None, mono=True)
        audio_data_list.append(waveform)
        
        # Clean up the temporary file
        os.remove(temp_audio_file_path)

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

# Define the Autoencoder class
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
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    epoch_loss = 0
    print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
    # epoch_energy = 0  # Energy in joules
    epoch_start_time = time.time()
    for batch_index, data in enumerate(dataloader):
        input_data = data[0]
        # Start power measurement
        # power_usage_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        # Forward pass
        output = model(input_data)
        loss = criterion(output, input_data)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        epoch_loss += batch_loss
        print(f"Batch {batch_index + 1}/{len(dataloader)}: Loss = {batch_loss:.6f}")
        # End power measurement
        # power_usage_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        # power_usage = power_usage_end - power_usage_start  # Energy in millijoules
        # epoch_energy += power_usage
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Duration in seconds
    print(f"Epoch duration in secods: {epoch_duration}")
    # Convert energy from millijoules to joules
    # epoch_energy_joules = epoch_energy / 1000
    average_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Complete: Average Loss = {average_loss:.6f}")

# Shutdown NVML
# pynvml.nvmlShutdown()
print("\nTraining complete!")

# Use the encoder to get latent representations
with torch.no_grad():
    latent_vectors = model.encoder(audio_tensor_flat)
    print("latent_vectors loaded")

# Generate a new latent vector by averaging and adding noise
new_latent = torch.mean(latent_vectors, dim=0, keepdim=True)
print("new latent defined")
noise = torch.randn_like(new_latent) * 0.1
print("noise defined")
new_latent += noise
print(f"new_latent = {new_latent}")

# Decode the new latent vector to get the new audio
new_audio = model.decoder(new_latent)
print("new audio defined")

# Rescale to original amplitude
new_audio = new_audio.detach().numpy()
new_audio = (new_audio + 1) / 2  # Scale back to [0, 1]
new_audio = new_audio * (audio_max - audio_min) + audio_min
print("rescaled to original amplitude")

# Convert the numpy array to a 1D array
new_audio = new_audio.squeeze()
print("conversion")

# Save the audio file
print("saving audio...")
output_file = 'new_audio.wav'
sf.write(output_file, new_audio, samplerate=sample_rate)

print("Finished!")
