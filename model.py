import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

class YamNetModel:
    def __init__(self):
        
        print("Loading YAMNet model...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("Model ready")

    def process_audio(self, file_path):
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Audio preprocessing
        if audio_data.dtype == np.int16:
            audio_data = audio_data / 32768.0
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resampling to 16kHz as required by YAMNet
        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = resample(audio_data, num_samples)
        
        # Extracting embeddings
        _, embeddings, _ = self.model(audio_data.astype(np.float32))
        return np.mean(embeddings, axis=0).tolist()

# Initialize the model instance
yamnet = YamNetModel()