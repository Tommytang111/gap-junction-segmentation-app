#Tommy Tang
#Sept 5, 2025
#Backend for GJS Web App using FastAPI

#Libraries
import os
import io
import base64
import zipfile
import shutil
from pathlib import Path
from typing import List
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import albumentations as A
import uvicorn
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.models import UNet, TestDataset
from src.inference import inference, visualize, evaluate
from src.utils import create_dataset_2d

#Initialize app
app = FastAPI(title="GJS Web App Backend", version="1.0", description="API for model inference")

#Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/models/unet_base_516imgs_sem_adult_8jkuifab.pt")
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

#Global model variable
model = None
augmentation = None

#Startup
@app.on_event("startup")
async def load_model():
    """Load the trained model at startup"""
    global model, augmentation
    try:
        model = UNet(classes=1).to(DEVICE)
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            
        #Augmentation
        augmentation = A.Compose([
            A.Normalize(mean=0.0, std=1.0),
            A.ToTensorV2()
        ])
    except Exception as e:
        print(f"Error loading model: {e}")

#Home page
@app.get("/")
async def home():
    """
    Home page with API information
    """
    return {
        "message": "GJS Web App Backend is running",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/upload-dataset": "Upload dataset for segmentation",
            "/run-inference": "Generates predictions on dataset",
            "/visualize": "XXX",
            "/evaluate": "XXX"
        }
    }
    
@app.post("/upload-dataset")
async def upload_dataset(files: List[UploadFile] = File(...)):
    """
    Upload a dataset of images and ground truth masks.
    Expects files to be organized as: images (*.png) and labels (*_label.png)
    """
    try:
        #Create session directory
        session_id = f"session_{len(os.listdir(UPLOAD_DIR))}"
        session_dir = UPLOAD_DIR / session_id
        imgs_dir = session_dir / "imgs"
        gts_dir = session_dir / "gts"
        
        imgs_dir.mkdir(parents=True, exist_ok=True)
        gts_dir.mkdir(parents=True, exist_ok=True)
    
        #Process uploaded files
        img_count = 0
        gt_count = 0
        
        for file in files:
            #Only process specific image formats
            if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            content = await file.read()
            
            if '_label' in file.filename:
                #This is a ground truth mask
                save_path = gts_dir / file.filename
                gt_count += 1
            else:
                #This is an image
                save_path = imgs_dir / file.filename
                img_count += 1
    
            with open(save_path, 'wb') as f:
                f.write(content)
    
        return {
            "session_id": session_id,
            "message": f"Dataset uploaded successfully",
            "image_count": img_count,
            "ground_truth_count": gt_count
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")
    
@app.post("/run-inference/{session_id}")
async def run_inference(session_id:str):
    """Run inference on uploaded dataset"""
    try:
        session_dir = UPLOAD_DIR / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        #Create output directory
        output_dir = OUTPUT_DIR / session_id
        output_dir.mkdir(exist_ok=True)
        
        #Define augmentation for inference
        valid_augmentation = A.Compose([
            A.Normalize(mean=0.0, std=1.0),
            A.ToTensorV2()
        ])

        #Run inference
        inference(
            model_path=MODEL_PATH,
            dataset=TestDataset,
            input_dir=str(session_dir),
            output_dir=str(output_dir),
            augmentation=valid_augmentation,
            filter=True
        )
        
        #Count results
        pred_files = list(output_dir.glob("*_pred.png"))
        
        return {
            "message": "Inference completed successfully",
            "predictions_count": len(pred_files),
            "output_directory": str(output_dir)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
@app.get("/visualize/{session_id}")
async def get_visualization(session_id:str, image_name:str=None):
    """Generate visualization plots for the results"""
    try:
        session_dir = UPLOAD_DIR / session_id
        output_dir = OUTPUT_DIR / session_id
        
        if not session_dir.exists() or not output_dir.exists():
            raise HTTPException(status_code=404, detail="Session or results not found")
        
        #Visualize
        fig = visualize(
            data_dir=str(session_dir),
            pred_dir=str(output_dir),
            base_name=image_name,
            style=1,
            random=image_name is None,
            gt=True
        )
        
        #Convert plot to base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return {"visualization": img_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")
    
@app.get("/evaluate/{session_id}")
async def get_evaluation(session_id:str):
    """Generate evaluation metrics and plots"""
    try:
        session_dir = UPLOAD_DIR / session_id
        output_dir = OUTPUT_DIR / session_id
        
        if not session_dir.exists() or not output_dir.exists():
            raise HTTPException(status_code=404, detail="Session or results not found")
        
        #Evaluate
        fig = evaluate(
            data_dir=str(session_dir),
            pred_dir=str(output_dir),
            title=f"Model Performance on {session_id}"
        )
        
        #Convert plot to base64
        img_buffer = io.BytersIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return {'evaluation_plot': img_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
@app.get("/sessions")
async def list_sessions():
    """List all available sessions"""
    sessions = [d.name for d in UPLOAD_DIR.iterdir() if d.is_dir()]
    return {"sessions": sessions}
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)