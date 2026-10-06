import os
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your categories
CLASS_NAMES = ['Potential Abnormality Detected', 'Normal Scan / No Tumor']

# Load pre-trained model weights (no training required)
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        prob_dict = {}
        for i, class_name in enumerate(CLASS_NAMES):
            prob_dict[class_name] = float(probabilities[i].item())
            
        predicted_idx = torch.argmax(probabilities).item()
        top_prediction = CLASS_NAMES[predicted_idx]
        
        return jsonify({
            'prediction': top_prediction,
            'probabilities': prob_dict
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)