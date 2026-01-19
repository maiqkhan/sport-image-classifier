from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from typing import List, Tuple

app = FastAPI(title="Sport Image Classifier")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load ONNX model
MODEL_PATH = "sport-image-classifier.onnx"
session = ort.InferenceSession(MODEL_PATH)

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Model input name: {input_name}")
print(f"Model input shape: {input_shape}")

# Sport class labels (adjust based on your model's training)
SPORT_CLASSES = [
    "air hockey",
"ampute football",
"archery",
"arm wrestling",
"axe throwing",
"balance beam",
"barell racing",
"baseball",
"basketball",
"baton twirling",
"bike polo",
"billiards",
"bmx",
"bobsled",
"bowling",
"boxing",
"bull riding",
"bungee jumping",
"canoe slamon",
"cheerleading",
"chuckwagon racing",
"cricket",
"croquet",
"curling",
"disc golf",
"fencing",
"field hockey",
"figure skating men",
"figure skating pairs",
"figure skating women",
"fly fishing",
"football",
"formula 1 racing",
"frisbee",
"gaga",
"giant slalom",
"golf",
"hammer throw",
"hang gliding",
"harness racing",
"high jump",
"hockey",
"horse jumping",
"horse racing",
"horseshoe pitching",
"hurdles",
"hydroplane racing",
"ice climbing",
"ice yachting",
"jai alai",
"javelin",
"jousting",
"judo",
"lacrosse",
"log rolling",
"luge",
"motorcycle racing",
"mushing",
"nascar racing",
"olympic wrestling",
"parallel bar",
"pole climbing",
"pole dancing",
"pole vault",
"polo",
"pommel horse",
"rings",
"rock climbing",
"roller derby",
"rollerblade racing",
"rowing",
"rugby",
"sailboat racing",
"shot put",
"shuffleboard",
"sidecar racing",
"ski jumping",
"sky surfing",
"skydiving",
"snow boarding",
"snowmobile racing",
"speed skating",
"steer wrestling",
"sumo wrestling",
"surfing",
"swimming",
"table tennis",
"tennis",
"track bicycle",
"trapeze",
"tug of war",
"ultimate",
"uneven bars",
"volleyball",
"water cycling",
"water polo",
"weightlifting",
"wheelchair basketball",
"wheelchair racing",
"wingsuit flying"

]

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for the ONNX model.
    Adjust this function based on your model's requirements.
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize (common normalization: ImageNet mean and std)
    # Adjust these values based on your model's training preprocessing
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array / 255.0 - mean) / std
    
    # Transpose to channel-first format (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension (1, C, H, W)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": session is not None}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify a sport image.
    Returns the predicted sport and confidence score.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run inference
        outputs = session.run(None, {input_name: processed_image})
        print(f"Number of outputs: {len(outputs)}")
        print(f"Model outputs shape: {[o.shape for o in outputs]}")
        
        # Handle different output formats
        predictions = outputs[0]
        if len(predictions.shape) > 1:
            predictions = predictions[0]  # Get first batch item if batched
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Number of classes in predictions: {len(predictions)}")
        print(f"Number of classes in SPORT_CLASSES: {len(SPORT_CLASSES)}")
        
        # Apply softmax to get probabilities
        probabilities = softmax(predictions)
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        print(f"Top indices: {top_indices}")
        print(f"Top probabilities: {probabilities[top_indices]}")
        
        results = []
        for idx in top_indices:
            class_idx = int(idx)
            # Only add if we have a label for this class
            if class_idx < len(SPORT_CLASSES):
                results.append({
                    "sport": SPORT_CLASSES[class_idx],
                    "confidence": float(probabilities[idx] * 100)
                })
            else:
                print(f"Warning: Class index {class_idx} exceeds SPORT_CLASSES length {len(SPORT_CLASSES)}")
        
        if not results:
            raise ValueError(f"No valid predictions generated. Model output has {len(predictions)} classes but SPORT_CLASSES has {len(SPORT_CLASSES)} labels.")
        
        return JSONResponse(content={
            "success": True,
            "prediction": results[0]["sport"],
            "confidence": round(results[0]["confidence"], 2),
            "top_predictions": results
        })
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
