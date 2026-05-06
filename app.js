//==========================================
// ISL SIGN LANGUAGE RECOGNITION - APP.JS
// ==========================================

// Configuration
const CONFIG = {
    API_BASE: 'http://localhost:5000',
    FRAME_INTERVAL: 500, // ms
    CONFIDENCE_THRESHOLD: 0.85,
    DEBOUNCE_COUNT: 3, // Same gesture 3x to confirm
    CANVAS_WIDTH: 640,
    CANVAS_HEIGHT: 480
};

// State Management
const state = {
    video: null,
    canvas: null,
    ctx: null,
    svg: null,
    currentText: '',
    lastPredictions: [],
    isWebcamActive: false,
    isProcessing: false,
    frameCount: 0
};

// DOM Elements
const elements = {
    prediction: document.getElementById('prediction'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidencePercent: document.getElementById('confidencePercent'),
    textOutput: document.getElementById('textOutput'),
    speakBtn: document.getElementById('speakBtn'),
    clearBtn: document.getElementById('clearBtn'),
    statusMessage: document.getElementById('statusMessage'),
    webcam: document.getElementById('webcam'),
    canvas: document.getElementById('canvas'),
    svg: document.getElementById('landmarks')
};

// ==========================================
// INITIALIZATION
// ==========================================

window.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing ISL Recognition System...');
    initWebcam();
    setupEventListeners();
});

function initWebcam() {
    // Set canvas size
    elements.canvas.width = CONFIG.CANVAS_WIDTH;
    elements.canvas.height = CONFIG.CANVAS_HEIGHT;
    
    state.video = elements.webcam;
    state.canvas = elements.canvas;
    state.ctx = state.canvas.getContext('2d');
    state.svg = elements.svg;

    // Request webcam access
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            width: { ideal: CONFIG.CANVAS_WIDTH },
            height: { ideal: CONFIG.CANVAS_HEIGHT }
        } 
    })
    .then(stream => {
        state.video.srcObject = stream;
        state.isWebcamActive = true;
        updateStatus('✓ Webcam active and ready', 'success');
        console.log('✓ Webcam stream started');
        
        // Start prediction loop
        startPredictionLoop();
    })
    .catch(error => {
        console.error('❌ Webcam Error:', error);
        updateStatus('❌ Webcam access denied. Check permissions.', 'error');
        if (error.name === 'NotAllowedError') {
            alert('Please allow webcam access in your browser settings.');
        }
    });
}

function setupEventListeners() {
    elements.speakBtn.addEventListener('click', speakText);
    elements.clearBtn.addEventListener('click', clearText);
}

// ==========================================
// PREDICTION LOOP
// ==========================================

function startPredictionLoop() {
    setInterval(() => {
        if (state.isWebcamActive && !state.isProcessing) {
            captureAndPredict();
        }
    }, CONFIG.FRAME_INTERVAL);
}

async function captureAndPredict() {
    state.isProcessing = true;
    state.frameCount++;

    try {
        // Draw video frame to canvas
        state.ctx.drawImage(state.video, 0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);
        
        // Convert canvas to base64
        const base64Image = state.canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        // Send to backend
        const response = await fetch(`${CONFIG.API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const result = await response.json();
        handlePredictionResult(result);

    } catch (error) {
        console.error('❌ Prediction Error:', error);
        updateStatus('⚠️ Backend not responding', 'warning');
    } finally {
        state.isProcessing = false;
    }
}

function handlePredictionResult(result) {
    if (!result.hand_detected) {
        updateStatus('No hand detected', 'warning');
        updatePredictionDisplay('--', 0);
        return;
    }

    const { label, confidence } = result;
    
    // Update UI with prediction
    updatePredictionDisplay(label, confidence);
    updateStatus('🟢 Hand detected - Processing...', 'success');

    // Debouncing logic
    handleDebounce(label, confidence);
}

function handleDebounce(label, confidence) {
    // Only process if confidence is above threshold
    if (confidence < CONFIG.CONFIDENCE_THRESHOLD) {
        return;
    }

    // Add to recent predictions
    state.lastPredictions.push(label);
    
    // Keep only last N predictions
    if (state.lastPredictions.length > CONFIG.DEBOUNCE_COUNT) {
        state.lastPredictions.shift();
    }

    // Check if last N predictions are the same
    if (state.lastPredictions.length === CONFIG.DEBOUNCE_COUNT) {
        const allSame = state.lastPredictions.every(p => p === label);
        
        if (allSame) {
            addToText(label);
            state.lastPredictions = []; // Reset debounce
        }
    }
}

function updatePredictionDisplay(label, confidence) {
    // Update label
    elements.prediction.textContent = label;

    // Update confidence bar
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceFill.style.width = confidencePercent + '%';
    elements.confidencePercent.textContent = confidencePercent + '%';

    // Color code confidence bar
    if (confidencePercent >= 85) {
        elements.confidenceFill.style.background = 'linear-gradient(90deg, #66BB6A 0%, #4CAF50 100%)';
    } else if (confidencePercent >= 70) {
        elements.confidenceFill.style.background = 'linear-gradient(90deg, #FFA726 0%, #FF9800 100%)';
    } else {
        elements.confidenceFill.style.background = 'linear-gradient(90deg, #EF5350 0%, #f44336 100%)';
    }
}

// ==========================================
// TEXT MANAGEMENT
// ==========================================

function addToText(gesture) {
    // Handle special gestures
    if (gesture === 'Space') {
        state.currentText += ' ';
    } else if (gesture === 'Backspace') {
        state.currentText = state.currentText.slice(0, -1);
    } else if (gesture === 'Clear') {
        state.currentText = '';
    } else {
        // Add gesture to text (space between words)
        if (state.currentText.length > 0 && state.currentText[state.currentText.length - 1] !== ' ') {
            state.currentText += ' ';
        }
        state.currentText += gesture;
    }

    updateTextDisplay();
    console.log('✓ Added gesture:', gesture);
}

function updateTextDisplay() {
    elements.textOutput.value = state.currentText.trim();
}

function clearText() {
    state.currentText = '';
    state.lastPredictions = [];
    updateTextDisplay();
    elements.prediction.textContent = '--';
    elements.confidenceFill.style.width = '0%';
    elements.confidencePercent.textContent = '0%';
    updateStatus('✓ Cleared all text', 'success');
    console.log('✓ Text cleared');
}

// ==========================================
// SPEECH SYNTHESIS
// ==========================================

async function speakText() {
    const text = state.currentText.trim();
    
    if (!text) {
        updateStatus('⚠️ No text to speak', 'warning');
        return;
    }

    // Disable button during speech
    elements.speakBtn.disabled = true;
    elements.speakBtn.innerHTML = '<span>⏳</span> Speaking...';

    try {
        // Send to backend for TTS
        const response = await fetch(`${CONFIG.API_BASE}/speak`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error('TTS Error');
        }

        // Get audio blob
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Play audio
        const audio = new Audio(audioUrl);
        audio.play();
        
        updateStatus('🔊 Speaking...', 'success');
        
        // Wait for audio to finish
        audio.onended = () => {
            elements.speakBtn.disabled = false;
            elements.speakBtn.innerHTML = '<span>🔊</span> Speak';
            updateStatus('✓ Speech complete', 'success');
        };

    } catch (error) {
        console.error('❌ Speech Error:', error);
        updateStatus('❌ Backend not responding', 'error');
        elements.speakBtn.disabled = false;
        elements.speakBtn.innerHTML = '<span>🔊</span> Speak';
    }
}

// ==========================================
// UI UPDATES
// ==========================================

function updateStatus(message, type = 'info') {
    const element = elements.statusMessage;
    element.textContent = message;
    element.className = 'status-message ' + type;
}

// ==========================================
// STARTUP CHECKS
// ==========================================

// Check if backend is available
async function checkBackend() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/health`);
        if (response.ok) {
            console.log('✓ Backend is available');
            return true;
        }
    } catch (error) {
        console.warn('⚠️ Backend not available yet:', error.message);
    }
    return false;
}

// Check browser compatibility
function checkBrowserCompatibility() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('❌ Your browser does not support webcam access. Please use Chrome, Firefox, or Edge.');
        return false;
    }
    return true;
}

// Run compatibility check on load
if (!checkBrowserCompatibility()) {
    updateStatus('❌ Browser not supported', 'error');
}

// Check backend availability
checkBackend().then(available => {
    if (!available) {
        updateStatus('⚠️ Backend may not be running on port 5000', 'warning');
    }
});

console.log('✓ App.js loaded successfully');
