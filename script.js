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
uploadButton.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag and Drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
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

// Classify image (placeholder - will connect to backend)
async function classifyImage(file) {
    // Show loader
    resultArea.style.display = 'block';
    loader.style.display = 'block';
    predictionSport.parentElement.style.display = 'none';
    
    // TODO: Replace this with actual API call to your backend
    // Example API call structure:
    /*
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        displayPrediction(data.sport, data.confidence);
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to classify image');
    }
    */
    
    // Simulated prediction for demo purposes
    setTimeout(() => {
        const sports = [
            'Basketball', 'Soccer', 'Tennis', 'Baseball', 
            'Football', 'Swimming', 'Volleyball', 'Hockey'
        ];
        const randomSport = sports[Math.floor(Math.random() * sports.length)];
        const randomConfidence = (Math.random() * 15 + 85).toFixed(1);
        
        displayPrediction(randomSport, randomConfidence);
    }, 1500);
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
