import cv2
import shutil
import os
from fastapi import FastAPI,UploadFile,File
from fastapi.responses import FileResponse
from ultralytics import YOLO
import torch

app = FastAPI()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('best.pt').to(device)

@app.post('/process_video')
async def process_video(file:UploadFile = File(...)):
    temp_input = f'temp_{file.filename}'
    output = f'processed_{file.filename}'
    with open(temp_input,'wb') as buffer:
        shutil.copyfileobj(file.file,buffer)
        
    out=None
    
    results = model(source=temp_input,stream=True,device=device,verbose=False,conf=.25)
    for result in results:
        if out is None:
            h,w = result.orig_shape
            fps =30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output,fourcc,fps,(w,h))
            
        annoted_frame = result.plot()
        out.write(annoted_frame)
        
    if out:
        out.release()
    
    os.remove(temp_input)
    return FileResponse(output,media_type='video/mp4',filename=output)
