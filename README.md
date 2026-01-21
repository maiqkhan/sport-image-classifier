# Sport Image Classifier - Deep Learning Application

## ⚠️ Disclaimer

This deep learning application is intended **for educational and learning purposes only**. It should **NOT** be used for commercial purposes without proper validation and testing.

**Important:**
- This project is a learning exercise in computer vision and deep learning
- The predictions made by this model are for demonstration purposes
- Model accuracy may vary depending on image quality and conditions
- Always verify predictions with domain expertise when needed

If you plan to use this application in production, please ensure proper testing and validation for your specific use case.

## Cloud Deployment
The Sport Image Classifier service is deployed in the cloud and available for testing here: [sport-image-classifier](https://sport-image-classifier-fzak.onrender.com/)

## Problem Statement

Sports classification from images is a challenging computer vision task with applications in sports analytics, automated sports broadcasting, content organization, and digital media management.

This project attempts to **classify images of different sports being played using deep learning techniques**. By analyzing visual features in images, the goal is to build an accurate predictive model that can **identify 100 different sports** from a single image, enabling automated sports content categorization and analysis.

## Dataset Overview

The dataset contains images across **100 different sport categories**:

**Sample Sport Categories:**
- Air hockey, Ampute football, Archery, Arm wrestling, Axe throwing
- Basketball, Baseball, Boxing, Bowling, BMX
- Cricket, Curling, Cheerleading
- Figure skating, Fencing, Football, Formula 1 racing
- Golf, Gymnastics (various events), Hockey, Horse racing
- Ice climbing, Javelin, Judo, Lacrosse
- Motorcycle racing, NASCAR racing, Olympic wrestling
- Pole vault, Polo, Rock climbing, Roller derby, Rowing, Rugby
- Skiing, Skateboarding, Snowboarding, Speed skating, Surfing, Swimming
- Tennis, Track cycling, Volleyball, Water polo, Weightlifting
- Wheelchair sports, Wingsuit flying, Wrestling
- And many more...

**Total Classes:** 100 sports categories

**Model Architecture:** Convolutional Neural Network (CNN) converted to ONNX format for efficient inference

## Project Goals

- Build a deep learning model capable of classifying 100 different sports from images
- Convert the trained model to ONNX format for optimized inference
- Achieve high accuracy across diverse sport categories
- Deploy the model as a web API service with an interactive front-end
- Provide user-friendly interface for real-time sport classification

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** that has been trained on sports image data and converted to ONNX format for deployment.

**Model Details:**
- **Input:** RGB images resized to 224x224 pixels
- **Format:** ONNX (Open Neural Network Exchange) for cross-platform compatibility
- **Preprocessing:** 
  - Image conversion to RGB
  - Resizing to 224x224 pixels
  - Normalization using ImageNet mean and standard deviation
  - Values: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Output:** Probability distribution across 100 sport classes

**Inference:**
The model uses ONNX Runtime for fast and efficient inference, making it suitable for real-time predictions.

## Web Application

The application provides a modern, interactive web interface for sport image classification:

**Features:**
- **Drag & Drop Upload:** Simply drag and drop an image or click to browse
- **Image Preview:** See your uploaded image before classification
- **Real-time Prediction:** Get instant classification results
- **Confidence Score:** View prediction confidence as a percentage
- **Responsive Design:** Works on desktop and mobile devices

**Tech Stack:**
- **Backend:** FastAPI (Python web framework)
- **Frontend:** HTML, CSS, JavaScript
- **ML Inference:** ONNX Runtime
- **Image Processing:** Pillow (PIL)

## Deployment

### Dependencies

To setup the development environment:
1. Clone/Fork this repository: `https://github.com/maiqkhan/sport-image-classifier.git`
2. Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Or create a virtual environment first:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Deploying the Prediction Service - Local Container

To deploy the FastAPI prediction service in a Docker container:

1. Build a Docker image using the Dockerfile in the root of the repository:
```bash
docker build -t sport-image-classifier:latest .
```

2. Once the Docker image is built, map port 8000 on the host, and deploy the container:
```bash
docker run -it -p 8000:8000 sport-image-classifier:latest
```

3. Open your browser and navigate to `http://localhost:8000` to access the web interface

### API Documentation

FastAPI provides interactive API documentation:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

**Available Endpoints:**

- `GET /` - Serves the web interface
- `POST /predict` - Upload an image and get sport classification
- `GET /health` - Health check endpoint
- `GET /classes` - Get list of all supported sport classes


### Deploying the Prediction Service - Cloud Deployment

To deploy the prediction service in the cloud, many options can be used:
- AWS Elastic Beanstalk
- DigitalOcean
- Render
- Heroku
- Google Cloud Run
- Azure App Service

**Example: Deploying to Render**

1. Create a Render account: [Render](https://dashboard.render.com/login)
2. Create a new Web Service using Public Git Repository to connect to this repository
3. Render will auto-detect the Dockerfile and prepare the environment to deploy the container in the cloud
   - Change the Region to whichever region is closest to you
   - Choose your preferred plan (Free tier available)
4. Deploy the web service and follow the generated link to your cloud-deployed application

**Note:** Ensure your ONNX model file is included in the repository or configure it to be downloaded during deployment.

## Usage Examples

### Web Interface

1. Navigate to the application URL
2. Click the upload area or drag & drop an image
3. Wait for the prediction (usually takes < 1 second)
4. View the predicted sport and confidence score

### API Usage (Python)

```python
import requests

url = 'http://localhost:8000/predict'

# Upload an image file
with open('sports_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
    prediction = response.json()
    
print(f"Predicted Sport: {prediction['sport']}")
print(f"Confidence: {prediction['confidence']:.2f}%")
print(f"Top 3 Predictions: {prediction['top_3']}")
```

### API Usage (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sports_image.jpg"
```

## Exploratory Data Analysis (EDA)

Comprehensive exploratory data analysis is available in the EDA directory:

- [Image Analysis.ipynb](EDA/Image%20Analysis.ipynb) - Complete image dataset analysis including:
  - Dataset download and structure exploration
  - Class distribution analysis (100 sport categories)
  - Image quality assessment (corrupted images, blur detection)
  - Color distribution and brightness analysis
  - Edge density and image complexity metrics
  - Sample visualization across all sport categories
  - Background variation and subject positioning analysis

### EDA Summary

The notebook includes detailed analysis of the sports image dataset:

- **Dataset Size:** 100 sport categories with varying numbers of images per class
- **Class Imbalance:** Significant imbalance detected (up to 4x difference between popular and rare sports)
- **Image Quality:** All images validated, corrupted files removed
- **Visual Characteristics:**
  - Dominant colors: Mix of outdoor greens, indoor venue colors, and athletic uniforms
  - Brightness distribution: Wide range from indoor low-light to outdoor bright scenes
  - Image complexity: Varies significantly by sport (team sports vs. individual sports)
  - Subject positioning: Most subjects relatively centered but with natural variation

**Key Findings:**
- The model needs to generalize well across classes with limited training data
- Sports with similar visual characteristics (e.g., different types of football) pose classification challenges
- Image preprocessing and normalization are critical due to varied lighting conditions

## Model Training

The training process uses transfer learning with EfficientNet-B0 as the base model:

### Training Script

[train/train.py](train/train.py) - Production-ready training script with:
- **Model Architecture:** EfficientNet-B0 with custom classification head
- **Transfer Learning:** Pre-trained ImageNet weights with frozen base model
- **Custom Classifier:** 
  - Dropout layers (configurable rate) for regularization
  - Dense layers (1280 → 512 → 100 classes)
  - ReLU activation functions
- **Data Augmentation:**
  - Random rotation (±10 degrees)
  - Random horizontal flipping
  - Color jittering (brightness, contrast, saturation)
  - Standard ImageNet normalization
- **Training Features:**
  - Early stopping with patience
  - Learning rate scheduling (ReduceLROnPlateau)
  - Best model checkpointing
  - ONNX export for deployment
  - Progress tracking with tqdm

### Hyperparameters

The final model uses the following configuration:
```python
{
    'weight_decay': 0.01,
    'lr': 0.001,
    'dropout_rate': 0.2,
    'batch_size': 128
}
```

**Training Details:**
- Optimizer: AdamW with weight decay
- Loss Function: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Early Stopping: Patience of 5 epochs
- Number of Epochs: 30 (with early stopping)

### Running Training

To train the model from scratch:

```bash
cd train
python train.py
```

The script will:
1. Load training and validation datasets from `../data/train` and `../data/valid`
2. Train the model with the specified hyperparameters
3. Save the best model checkpoint as `best_model.pth`
4. Export the model to ONNX format as `best_sport-image-classifier-model.onnx`
5. Display training progress and metrics

**Note:** Training requires a GPU for reasonable training times. The script automatically detects and uses CUDA if available.

### Training Notebook

For experimentation and hyperparameter tuning, see:
- [train/train.ipynb](train/train.ipynb) - Jupyter notebook with:
  - Grid search implementation for hyperparameter tuning
  - Visualization of training metrics
  - Model evaluation on test set
  - ONNX model validation

## Project Structure

```
sport-image-classifier/
│
├── main.py                          # FastAPI application
├── sport-image-classifier.onnx      # Trained ONNX model
├── pyproject.toml                   # Project dependencies (uv)
├── requirements.txt                 # Python dependencies (pip)
├── Dockerfile                       # Container configuration
├── README.md                        # Project documentation
│
├── EDA/                             # Exploratory Data Analysis
│   └── Image Analysis.ipynb         # Comprehensive image dataset analysis
│
├── train/                           # Model Training
│   ├── train.py                     # Production training script
│   └── train.ipynb                  # Training experimentation notebook
│
├── data/                            # Dataset directory (gitignored)
│   ├── train/                       # Training images
│   ├── valid/                       # Validation images
│   └── test/                        # Test images
│
├── static/                          # Frontend assets
│   ├── script.js                    # JavaScript functionality
│   └── style.css                    # CSS styling
│
├── templates/                       # HTML templates
│   └── index.html                   # Main web interface
│
└── __pycache__/                     # Python cache (auto-generated)
```

## Future Improvements

- [ ] Add model training notebooks showing the CNN architecture and training process
- [ ] Implement batch prediction for multiple images
- [ ] Add more detailed error handling and logging
- [ ] Implement model performance metrics and confusion matrix
- [ ] Create dataset preprocessing and augmentation documentation
- [ ] Implement A/B testing for model versions

## Technologies Used

- **Python 3.x**
- **FastAPI** - Modern, fast web framework
- **ONNX Runtime** - High-performance ML inference
- **Pillow (PIL)** - Image processing
- **NumPy** - Numerical computations
- **Uvicorn** - ASGI server
- **HTML/CSS/JavaScript** - Frontend interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational purposes. Please add appropriate license information based on your requirements.

## Acknowledgments

- Dataset and model training details:  [100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- ONNX Runtime for efficient inference
- FastAPI for the excellent web framework

## Contact

For questions or feedback, please open an issue in the repository.
