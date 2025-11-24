import torch
from torch import nn
import joblib
from torchvision.transforms import transforms
from PIL import Image

class MyModule(nn.Module):
        def __init__(self,label_encoder,input_dim=3):
            super().__init__()
            self.sequential = nn.Sequential(
                nn.Conv2d(input_dim,32,kernel_size=3,padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                
                nn.Conv2d(32,64,kernel_size=3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                
                nn.Conv2d(64,128,kernel_size=3,padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2,2),

                nn.Flatten(),
                nn.Linear((128*4*4),512),
                nn.ReLU(),
                nn.Dropout(.3),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Dropout(.3),
                nn.Linear(256,len(label_encoder.classes_))

            )
        def forward(self,x):
            x = self.sequential(x)
            return x

def load_models(model_path,label_encoder_path):
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)

    transform =  transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN,NORM_STD)]
)   
    try:
        label_encoer = joblib.load(label_encoder_path)
    except FileNotFoundError:
        print("Label_encoder path not found")
        return None
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    model = MyModule(label_encoer).to(DEVICE)
    model.load_state_dict(torch.load(model_path,map_location=DEVICE,weights_only=True))
    model.eval()
    return model,label_encoer,transform,DEVICE

def process_image(input,model,label_encoder,transform,device):
    image=transform(input).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output,1)
        predicted_index = torch.argmax(probs,1).item()  
        confidence = probs[0,predicted_index].item()
        label = label_encoder.inverse_transform([predicted_index])[0]
        return {'Prediction':label,'Confidence':f"{confidence*100:.2f}%"}


if __name__ == '__main__':
     pass