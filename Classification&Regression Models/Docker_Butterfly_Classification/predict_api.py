from fastapi import FastAPI, UploadFile, File
import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import nn
import joblib

app = FastAPI()


label_encoder = joblib.load('label_encoder.joblib')

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

model = models.resnet18()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 75)
)
model.load_state_dict(torch.load("best_model_butteryfly.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image = Image.open(BytesIO(await file.read())).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        preds = torch.argmax(output,axis=1).item()
        probs = torch.softmax(output,axis=1).cpu().numpy().flatten()
        top_n=3
        top_indicides =probs.argsort()[-top_n:][::-1]
        top_labels = label_encoder.inverse_transform(top_indicides)
        top_preds = [(label,float(probs[x])) for label,x in zip(top_labels,top_indicides)]
            
        print(top_preds)
        prediction= label_encoder.inverse_transform([preds])[0]
        return {'Prediction':prediction}