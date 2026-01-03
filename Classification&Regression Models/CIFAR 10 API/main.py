from process_image import process_image,load_models
from fastapi import FastAPI,UploadFile,File
import uvicorn
import os
from io import BytesIO
from PIL import Image
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

model,label_encoder,transform,device = None,None,None,None
model_path = './CIFAR 10 API/best_model_CIFAR.pth'
label_encoder_path = './CIFAR 10 API/labelencoder.joblib'

@asynccontextmanager
async def startup(app:FastAPI):
    print('Loading starting assests')
    global model,label_encoder,transform,device
    model,label_encoder,transform,device=load_models(model_path,label_encoder_path)
    yield

app =FastAPI(title='CIFAR-10 Predictions',lifespan=startup)

@app.post("/predicton",tags=['Prediction'])
async def predict_image(file: UploadFile = File(...,description='Pass image for prediction')):
    image = await file.read()
    image = Image.open(BytesIO(image)).convert('RGB')
    output = process_image(image,model,label_encoder,transform,device)
    return JSONResponse(content=(output))


if __name__=='__main__':

    uvicorn.run(app,host='0.0.0.0',port=8000)