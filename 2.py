from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ghostnet import GhostNet
import base64
import numpy as np
import shutil
import torch
import torchvision.transforms as transforms
from ghostnet import GhostNet
import tempfile
app = Flask(__name__)
CORS(app)

# Define the /detect route
@app.route('/detect', methods=['POST'])
def detect_audio():
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    temp_filename = 'temp_audio_file.mp3'
    audio_file.save(temp_filename)

    try:
        
        filename = 'temp_audio_file.mp3'
        output_folder = 'proxynas_test'

        os.makedirs(output_folder, exist_ok=True)


        audio_path = filename
        y, sr = librosa.load(audio_path, sr=16000)  # Resample to 16 kHz

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224, fmax=sr//2)

        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=224/5)  # Ensures the output size is 224x224 pixels
        librosa.display.specshow(S_dB, sr=sr, ax=ax, cmap='viridis')

        ax.axis('off')

        output_image_path = os.path.join(output_folder, filename.replace('.flac', '') + '.png')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        img = Image.open(output_image_path).convert('RGB')

        img = img.resize((224, 224))

        img.save(output_image_path)

        # Testing single image
        def test_single_image(model_path, image_path, transform, device):
            # Load the saved model
            model = GhostNet(num_classes=2).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))  # Ensures correct device mapping
            model.eval()

            # Preprocess and test the single image
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)

            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            confidence_fake = probabilities[0]
            confidence_real = probabilities[1]

            print(f"Fake: {confidence_fake:.4f}, Real: {confidence_real:.4f}")
            return confidence_real,confidence_fake

        # Test single image

        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        single_image_path = 'proxynas_test/temp_audio_file.mp3.png'
        model_save_path = "ghostnet_modele.pkl"
        real_confidence,fake_confidence=test_single_image(model_save_path, single_image_path, transform, device)

        
        is_fake = ""
        if (real_confidence>fake_confidence):
            is_fake=0
        else:
            is_fake=1
        
        print(is_fake,real_confidence,fake_confidence)
        
        real_confidence =  np.float32(round(real_confidence,5))
        fake_confidence =  np.float32(round(fake_confidence,5))
        
        
        with open("proxynas_test/temp_audio_file.mp3.png", "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        response = {
            'is_fake': is_fake, 
            'real_confidence': float(real_confidence),
            'fake_confidence': float(fake_confidence),
            'melspec_image':encoded_image
        }
        
        
        # response={}
        # print(response)
        
        return jsonify({'data':response})
    
    finally:
        if os.path.exists(temp_filename):
            print("Hellooooo")
            os.remove(temp_filename)
            shutil.rmtree("proxynas_test")
        
        
        
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
