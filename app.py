from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import io
import base64
import os

# --- Model Structure ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

app = Flask(__name__)

# --- Load Model ---
device = torch.device("cpu")
model = Net().to(device)
MODEL_PATH = "mnist_cnn.pt"

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded.")
    except:
        print("Error loading model.")
else:
    print("WARNING: Model file not found. Run train_model.py first.")

# --- Attack Logic ---
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# --- Helper ---
def process_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    # Invert if image is black-on-white (common for user uploads)
    if ImageOps.grayscale(image).getpixel((0,0)) > 128:
        image = ImageOps.invert(image)
    return transform(image).unsqueeze(0)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'})
    
    file = request.files['file']
    try: epsilon = float(request.form.get('epsilon', 0.1))
    except: epsilon = 0.1
    
    # 1. Process
    img_bytes = file.read()
    data = process_image(img_bytes).to(device)
    data.requires_grad = True

    # 2. Predict
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1]
    init_probs = torch.exp(output).detach().numpy().tolist()[0]

    # 3. Attack
    loss = F.nll_loss(output, init_pred[0])
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = fgsm_attack(data, epsilon, data_grad)

    # 4. Final Predict
    final_output = model(perturbed_data)
    final_pred = final_output.max(1, keepdim=True)[1]
    final_probs = torch.exp(final_output).detach().numpy().tolist()[0]

    return jsonify({
        'original_class': int(init_pred.item()),
        'adversarial_class': int(final_pred.item()),
        'original_confidence': init_probs,
        'adversarial_confidence': final_probs
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
