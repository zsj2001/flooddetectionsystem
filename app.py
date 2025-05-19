from flask import Flask, render_template, request, redirect
import os
import base64
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load class names from classes.txt
with open('classes.txt', 'r') as file:
    rows = file.readlines()
rows = [row.strip() for row in rows]
num_classes = len(rows)

# Define the network architecture
class TransferLearningCNN(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningCNN, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2()
        in_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet_v2(x)

# Load the trained model
trainedModel = 'best.pt'
model = TransferLearningCNN(num_classes)
model.load_state_dict(torch.load(trainedModel, map_location=torch.device("cpu")))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    input_image = image.convert('RGB')
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_index = torch.argmax(probabilities).item()
    confidence_score = probabilities[predicted_index].item()
    predicted_class = rows[predicted_index]
    return predicted_class, confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_data = None
    error = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "No file part in the request."
            return render_template('index.html', error=error)
        
        file = request.files['file']
        
        if file.filename == '':
            error = "No file selected."
            return render_template('index.html', error=error)
        
        if not allowed_file(file.filename):
            error = "Invalid file type. Please upload an image file (png, jpg, jpeg, gif)."
            return render_template('index.html', error=error)
        
        try:
            image_bytes = file.read()
            image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            error = "Error processing image file."
            return render_template('index.html', error=error)
        
        predicted_class, confidence_score = predict(image)
        prediction = predicted_class
        confidence = confidence_score
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
    return render_template('index.html', prediction=prediction, confidence=confidence, image_data=image_data, error=error)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
