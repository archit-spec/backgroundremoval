# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from celery import Celery
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import base64
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import Response
import asyncio
from celery.result import AsyncResult
import logging

# set up logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Create Celery app
celery = Celery('tasks', 
                broker='redis://localhost:6379/0',
                backend='redis://localhost:6379/1')
celery.conf.update(
    result_backend='redis://localhost:6379/1',
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)
# Load the ONNX model
session = ort.InferenceSession('model.onnx')

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Semaphore to limit concurrent requests
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as file:
        return file.read()

@celery.task(bind=True)
def process_image(self, image_bytes):
    # Simulate progress updates
    self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100})
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize the image to the expected dimensions (e.g., 1024x1024)
    target_size = (1024, 1024)
    image = image.resize(target_size, Image.LANCZOS)

    self.update_state(state='PROGRESS', meta={'current': 25, 'total': 100})

    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # Change to CHW format
    img_array = img_array[None, :]  # Add batch dimension

    self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100})

    # Run inference
    inputs = {session.get_inputs()[0].name: img_array}
    outputs = session.run(None, inputs)
    
    self.update_state(state='PROGRESS', meta={'current': 75, 'total': 100})

    # Handle the output based on its shape
    result = outputs[0].squeeze()
    if len(result.shape) == 2:
        # If the output is a single channel (alpha mask)
        mask = (result * 255).astype(np.uint8)
        # Apply the mask to the original image
        image = np.array(image)
        image = np.concatenate([image, mask[:,:,np.newaxis]], axis=2)
        result_image = Image.fromarray(image, mode='RGBA')
    elif len(result.shape) == 3:
        # If the output is already an RGB image with background removed
        result = (result * 255).astype(np.uint8)
        result_image = Image.fromarray(result, mode='RGB')
    else:
        raise ValueError("Unexpected output format from the model")

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    
    self.update_state(state='PROGRESS', meta={'current': 100, 'total': 100})
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/remove-background")
@limiter.limit("5/minute")
async def remove_background(request: Request, response: Response, files: List[UploadFile] = File(...)):
    async with semaphore:
        tasks = []
        try:
            for file in files:
                # Check file size (limit to 5MB)
                content = await file.read()
                if len(content) > 5_000_000:
                    raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 5MB limit")
                
                task = process_image.delay(content)
                tasks.append((file.filename, task.id))
            
            return {"tasks": tasks}
        except Exception as e:
            logger.error(f"Error in remove_background: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/task/{task_id}")
@limiter.limit("20/minute")
async def get_task_result(request: Request, response: Response, task_id: str):
    try:
        task = AsyncResult(task_id, app=celery)
        if task.ready():
            return {"status": "completed", "result": task.result}
        else:
            return {"status": "processing", "current": task.info.get('current', 0), "total": task.info.get('total', 100)}
    except Exception as e:
        logger.error(f"Error retrieving task result: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving task result")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)