import os
import requests
import dropbox
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import random
import tempfile
from dotenv import load_dotenv
import time
import subprocess
from codecarbon import EmissionsTracker
import sounddevice as sd

# Load environment variables from .env file
print("Loading .env variables...")
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
AUTHORIZATION_CODE = input("Enter the authorization code: ")

# Step 2: Exchange the authorization code for access and refresh tokens
data = {
    'code': AUTHORIZATION_CODE,
    'grant_type': 'authorization_code',
    'redirect_uri': REDIRECT_URI,
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET
}

response = requests.post(TOKEN_URL, data=data)
response_data = response.json()
access_token = response_data.get('access_token')
refresh_token = response_data.get('refresh_token')

print("Access Token:", access_token)
print("Refresh Token:", refresh_token)

# Initialize Dropbox client with the access token
print("Initialising Dropbox client with access token")
dbx = dropbox.Dropbox(access_token)
dropbox_folder_path = '/audio'

# List all files in the Dropbox folder
print("listing all files now...")
file_entries = dbx.files_list_folder(dropbox_folder_path).entries

# Filter out only the .mp3 files
print("I'm selecting the mp3 files...")
mp3_files = [entry for entry in file_entries if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.mp3')]

# Randomly select 10 audio files from the list
selected_files = random.sample(mp3_files, min(10, len(mp3_files)))

# Download, trim, and save the selected audio files temporarily
audio_data_list = []
temp_audio_paths = []

print("now I am downloading and trimming the files...")
for entry in selected_files:
    dropbox_path = f"{dropbox_folder_path}/{entry.name}"
    _, res = dbx.files_download(dropbox_path)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(res.content)
        temp_audio_file_path = temp_audio_file.name
    
    # Load the audio and trim silence
    waveform, sample_rate = librosa.load(temp_audio_file_path, sr=None, mono=True)
    trimmed_waveform, _ = librosa.effects.trim(waveform)  # Trim leading and trailing silence

    # Save the trimmed audio to a new temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as trimmed_audio_file:
        sf.write(trimmed_audio_file.name, trimmed_waveform, sample_rate)
        temp_audio_paths.append(trimmed_audio_file.name)
    
    # Append the trimmed waveform for training
    audio_data_list.append(trimmed_waveform)
    
    # Remove the original temporary file
    os.remove(temp_audio_file_path)

print("ugh... That was some internet usage eheh... Now I'm normalising and converting audio waveforms to tensor")
# Normalize and convert to PyTorch tensor
min_length = min([len(waveform) for waveform in audio_data_list])
audio_data = np.stack([waveform[:min_length] for waveform in audio_data_list])
audio_tensor = torch.from_numpy(audio_data).float()
audio_min = audio_tensor.min().item()
audio_max = audio_tensor.max().item()
audio_tensor = (audio_tensor - audio_min) / (audio_max - audio_min)
audio_tensor = audio_tensor * 2 - 1

print("ok, now just going to define the autoencoder class")
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
print("...and flattening the audio data for training...")
input_dim = audio_tensor.shape[1]
audio_tensor_flat = audio_tensor

# Create a DataLoader
print("...creating a dataloadeeeer...")
dataset = torch.utils.data.TensorDataset(audio_tensor_flat)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model, loss function, and optimizer
print("initialising the model, the loss function and the optimiser...")
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print("oh, yeah, let me also define the gpu_power_usage class")
def get_gpu_power_usage():
    # Get power usage for an NVIDIA GPU (in watts)
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True
    )
    power_usage_watts = float(result.stdout.strip())
    return power_usage_watts

print("aaand I'm also starting the emissions tracker...")
tracker = EmissionsTracker()
tracker.start()

print("OK. Set! Training loop starts now.")
# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    epoch_loss = 0
    print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
    epoch_start_time = time.time()

    batch_power_usages = []
    for batch_index, data in enumerate(dataloader):
        # Measure power before starting the batch
        power_before = get_gpu_power_usage()

        input_data = data[0]
        output = model(input_data)
        loss = criterion(output, input_data)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure power after batch
        power_after = get_gpu_power_usage()

        # Estimate average power for this batch
        avg_power = (power_before + power_after) / 2
        batch_power_usages.append(avg_power)

        batch_loss = loss.item()
        epoch_loss += batch_loss

        print(f"Batch {batch_index + 1}/{len(dataloader)}: Loss = {batch_loss:.6f}, Power = {avg_power:.2f} Watts")

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # Duration in seconds
    print(f"Epoch duration in secods: {epoch_duration}")
    average_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Complete: Average Loss = {average_loss:.6f}")

print("\nTraining complete!")
tracker.stop()

# Use the encoder to get latent representations
print("just getting the latent representation")
with torch.no_grad():
    latent_vectors = model.encoder(audio_tensor_flat)
    print("latent_vectors loaded")

# Generate a new latent vector by averaging and adding noise
print("and now generating a new (f)lat(ul)ent vector")
new_latent = torch.mean(latent_vectors, dim=0, keepdim=True)
print("new latent defined")
noise = torch.randn_like(new_latent) * 0.1
print("noise defined")
new_latent += noise
print(f"new_latent = {new_latent}")

print("decoding it back to audiooo. Oh, how exciting")
# Decode the new latent vector to get the new audio
new_audio = model.decoder(new_latent)

print("let me rescale it to original amplitudeeee")
# Rescale to original amplitude
new_audio = new_audio.detach().numpy()
new_audio = (new_audio + 1) / 2  # Scale back to [0, 1]
new_audio = new_audio * (audio_max - audio_min + 1e-7) + audio_min

# Convert the numpy array to a 1D array
new_audio = new_audio.squeeze()

# Save the audio file
print("saving it!")
output_file = 'new_audio.wav'
sf.write(output_file, new_audio, samplerate=sample_rate)

print("Finished!")