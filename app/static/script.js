// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const clearButton = document.getElementById('clearButton');
const resultArea = document.getElementById('resultArea');
const predictionSport = document.getElementById('predictionSport');
const predictionConfidence = document.getElementById('predictionConfidence');
const loader = document.getElementById('loader');

// Click to upload
uploadButton.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

uploadArea.addEventListener('click', (e) => {
    if (e.target.id !== 'uploadButton') {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and Drop functionality
uploadArea.addEventListener('dragenter', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        alert('Please drop an image file');
    }
});

// Clear button
clearButton.addEventListener('click', () => {
    resetUI();
});

// Handle file upload
function handleFile(file) {
    // Display image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        
        // Simulate prediction (replace with actual API call)
        classifyImage(file);
    };
    reader.readAsDataURL(file);
}

// Classify image using backend API
async function classifyImage(file) {
    // Show loader
    resultArea.style.display = 'block';
    loader.style.display = 'block';
    predictionSport.parentElement.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Classification request failed');
        }
        
        const data = await response.json();
        displayPrediction(data.prediction, data.confidence);
    } catch (error) {
        console.error('Error:', error);
        loader.style.display = 'none';
        alert(`Failed to classify image. Error: ${error.message}`);
    }
}

// Display prediction results
function displayPrediction(sport, confidence) {
    loader.style.display = 'none';
    predictionSport.parentElement.style.display = 'block';
    predictionSport.textContent = sport;
    predictionConfidence.textContent = `${confidence}% confidence`;
}

// Reset UI
function resetUI() {
    uploadArea.style.display = 'block';
    previewArea.style.display = 'none';
    resultArea.style.display = 'none';
    previewImage.src = '';
    fileInput.value = '';
}
