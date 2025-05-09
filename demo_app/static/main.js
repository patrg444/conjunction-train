/**
 * EmotionTrack - Marketing Analytics Platform
 * Main JavaScript for the application
 */

// Global variables
let webgazerInitialized = false;
let gazeData = [];
let emotionData = [];
let timeSeriesChart = null;
let isRunning = false;
let frameCount = 0;
let lastFrameTime = 0;
let targetFPS = 15; // Limit to 15 FPS to match model training
let frameInterval = 1000 / targetFPS;
let calibrationPoints = [];
let calibrationInProgress = false;

// Emotion colors (matching the backend)
const emotionColors = {
    'Anger': 'rgb(255, 0, 0)',    // Red
    'Disgust': 'rgb(255, 140, 0)', // Orange
    'Fear': 'rgb(255, 255, 0)',    // Yellow
    'Happy': 'rgb(0, 255, 0)',     // Green
    'Neutral': 'rgb(0, 255, 255)', // Cyan
    'Sad': 'rgb(0, 0, 255)'        // Blue
};

// DOM elements
const webcamElement = document.getElementById('webcam');
const eyeTrackingOverlay = document.getElementById('eyeTrackingOverlay');
const gazeDot = document.getElementById('gazeDot');
const emotionBars = document.getElementById('emotionBars');
const emotionLabel = document.getElementById('emotionLabel');
const engagementScore = document.getElementById('engagementScore');
const attentionDuration = document.getElementById('attentionDuration');
const emotionalImpact = document.getElementById('emotionalImpact');
const heatmapOverlay = document.getElementById('heatmapOverlay');
const emotionTimeChart = document.getElementById('emotionTimeChart');
const calibrationModal = document.getElementById('calibrationModal');
const calibrateBtn = document.getElementById('calibrateBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const startCalibrationBtn = document.getElementById('startCalibrationBtn');
const finishCalibrationBtn = document.getElementById('finishCalibrationBtn');
const calibrationPointsContainer = document.getElementById('calibrationPoints');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    setupEventListeners();
    initializeCharts();
});

// Initialize UI components
function initializeUI() {
    // Set up webcam overlay canvas
    eyeTrackingOverlay.width = webcamElement.clientWidth;
    eyeTrackingOverlay.height = webcamElement.clientHeight;

    // Set up heatmap overlay canvas
    const marketingContent = document.getElementById('marketingContent');
    heatmapOverlay.width = marketingContent.clientWidth;
    heatmapOverlay.height = marketingContent.clientHeight;

    // Initialize heatmap
    heatmapInstance = createHeatmap({
        canvas: heatmapOverlay,
        radius: 25,
        maxOpacity: 0.6,
        minOpacity: 0,
        blur: 0.85
    });

    // Create empty emotion bars for each emotion
    Object.keys(emotionColors).forEach(emotion => {
        createEmotionBar(emotion, 0);
    });
}

// Set up event listeners
function setupEventListeners() {
    // Calibration button
    calibrateBtn.addEventListener('click', openCalibrationModal);
    
    // Start button
    startBtn.addEventListener('click', startAnalysis);
    
    // Stop button
    stopBtn.addEventListener('click', stopAnalysis);
    
    // Calibration controls
    startCalibrationBtn.addEventListener('click', startCalibration);
    finishCalibrationBtn.addEventListener('click', finishCalibration);
    
    // Window resize event
    window.addEventListener('resize', handleResize);
}

// Initialize Charts.js for time series data
function initializeCharts() {
    const ctx = emotionTimeChart.getContext('2d');
    
    timeSeriesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: Object.keys(emotionColors).map(emotion => ({
                label: emotion,
                data: [],
                borderColor: emotionColors[emotion],
                backgroundColor: `${emotionColors[emotion]}33`, // Light opacity
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.4
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Probability'
                    },
                    ticks: {
                        callback: value => `${Math.round(value * 100)}%`
                    }
                }
            }
        }
    });
}

// Handle window resize
function handleResize() {
    // Update webcam overlay dimensions
    eyeTrackingOverlay.width = webcamElement.clientWidth;
    eyeTrackingOverlay.height = webcamElement.clientHeight;
    
    // Update heatmap dimensions
    const marketingContent = document.getElementById('marketingContent');
    heatmapOverlay.width = marketingContent.clientWidth;
    heatmapOverlay.height = marketingContent.clientHeight;
    
    // Redraw heatmap with current data
    drawHeatmap();
}

// Initialize WebGazer
async function initializeWebGazer() {
    if (webgazerInitialized) return;
    
    try {
        // Set up WebGazer with prediction callback
        await webgazer
            .setGazeListener((data, timestamp) => {
                if (data == null || !isRunning) return;
                
                // Update gaze dot position
                updateGazeDot(data.x, data.y);
                
                // Add data to gaze array
                gazeData.push({
                    x: data.x,
                    y: data.y,
                    timestamp: Date.now()
                });
                
                // Update heatmap periodically
                if (gazeData.length % 5 === 0) {
                    updateHeatmap();
                }
            })
            .begin();
        
        // Wait for camera permission and initialization
        while (!webgazer.isReady()) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Set up WebGazer settings
        webgazer.showPredictionPoints(false);
        webgazer.showVideo(false);
        
        webgazerInitialized = true;
        console.log('WebGazer initialized successfully');
    } catch (error) {
        console.error('Error initializing WebGazer:', error);
        alert('Failed to initialize eye tracking. Please check console for details.');
    }
}

// Open calibration modal
function openCalibrationModal() {
    calibrationModal.style.display = 'flex';
    generateCalibrationPoints();
}

// Generate calibration points in a grid
function generateCalibrationPoints() {
    calibrationPointsContainer.innerHTML = '';
    calibrationPoints = [];
    
    // Create a 3x3 grid of calibration points
    const rows = 3;
    const cols = 3;
    
    const containerWidth = calibrationPointsContainer.clientWidth;
    const containerHeight = calibrationPointsContainer.clientHeight;
    
    const paddingX = containerWidth * 0.1;
    const paddingY = containerHeight * 0.1;
    
    const stepX = (containerWidth - 2 * paddingX) / (cols - 1);
    const stepY = (containerHeight - 2 * paddingY) / (rows - 1);
    
    // Create each point
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const x = paddingX + col * stepX;
            const y = paddingY + row * stepY;
            
            const point = document.createElement('div');
            point.className = 'calibration-point';
            point.style.position = 'absolute';
            point.style.left = `${x}px`;
            point.style.top = `${y}px`;
            point.style.width = '20px';
            point.style.height = '20px';
            point.style.borderRadius = '50%';
            point.style.backgroundColor = 'rgba(66, 133, 244, 0.8)';
            point.style.transform = 'translate(-50%, -50%)';
            point.style.display = 'none';
            
            // Add to DOM and point array
            calibrationPointsContainer.appendChild(point);
            calibrationPoints.push({
                element: point,
                x,
                y,
                clicked: false
            });
        }
    }
}

// Start calibration process
function startCalibration() {
    if (calibrationInProgress) return;
    
    calibrationInProgress = true;
    startCalibrationBtn.disabled = true;
    
    // Ensure WebGazer is initialized
    initializeWebGazer().then(() => {
        // Clear existing calibration
        webgazer.clearData();
        
        // Display calibration points one by one
        let currentPointIndex = 0;
        
        function showNextPoint() {
            if (currentPointIndex >= calibrationPoints.length) {
                // Calibration complete
                calibrationInProgress = false;
                finishCalibrationBtn.disabled = false;
                return;
            }
            
            // Hide all points
            calibrationPoints.forEach(point => {
                point.element.style.display = 'none';
            });
            
            // Show current point
            const currentPoint = calibrationPoints[currentPointIndex];
            currentPoint.element.style.display = 'block';
            
            // Add click handler
            currentPoint.element.onclick = () => {
                // Call WebGazer click method for calibration
                const boundingRect = calibrationPointsContainer.getBoundingClientRect();
                const clickX = boundingRect.left + currentPoint.x;
                const clickY = boundingRect.top + currentPoint.y;
                
                webgazer.recordScreenPosition(clickX, clickY, currentPoint.x, currentPoint.y);
                
                currentPoint.clicked = true;
                currentPointIndex++;
                
                // Show next point after delay
                setTimeout(showNextPoint, 500);
            };
        }
        
        // Start showing points
        showNextPoint();
    });
}

// Finish calibration
function finishCalibration() {
    // Close modal
    calibrationModal.style.display = 'none';
    
    // Reset buttons
    finishCalibrationBtn.disabled = true;
    startCalibrationBtn.disabled = false;
    
    // Report calibration quality
    const calibrationComplete = calibrationPoints.every(point => point.clicked);
    
    if (calibrationComplete) {
        console.log('Calibration completed successfully');
    } else {
        console.warn('Calibration incomplete - some points were not clicked');
    }
}

// Start the analysis process
async function startAnalysis() {
    if (isRunning) return;
    
    // Start webcam if not already
    try {
        if (!webcamElement.srcObject) {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: true, 
                audio: true 
            });
            webcamElement.srcObject = stream;
            await new Promise(resolve => {
                webcamElement.onloadedmetadata = resolve;
            });
        }
        
        // Initialize WebGazer if not already
        await initializeWebGazer();
        
        // Start analysis loop
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        calibrateBtn.disabled = true;
        
        // Show gaze dot
        gazeDot.style.display = 'block';
        
        // Start processing frames
        lastFrameTime = Date.now();
        requestAnimationFrame(processFrame);
        
        console.log('Analysis started');
    } catch (error) {
        console.error('Error starting analysis:', error);
        alert('Failed to access webcam. Please check permissions and try again.');
    }
}

// Stop the analysis process
function stopAnalysis() {
    if (!isRunning) return;
    
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    calibrateBtn.disabled = false;
    
    // Hide gaze dot
    gazeDot.style.display = 'none';
    
    console.log('Analysis stopped');
}

// Process a video frame for emotion recognition
async function processFrame() {
    if (!isRunning) return;
    
    const currentTime = Date.now();
    const elapsed = currentTime - lastFrameTime;
    
    // Throttle to target FPS
    if (elapsed >= frameInterval) {
        lastFrameTime = currentTime - (elapsed % frameInterval);
        frameCount++;
        
        // Capture current frame
        const imageData = captureFrame();
        
        // Process frame
        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            if (response.ok) {
                const result = await response.json();
                updateEmotionDisplay(result);
                updateEngagementMetrics(result);
                updateTimeSeriesChart(result);
            } else {
                console.error('Error processing frame:', await response.text());
            }
        } catch (error) {
            console.error('Network error:', error);
        }
    }
    
    // Continue animation loop
    requestAnimationFrame(processFrame);
}

// Capture current webcam frame as base64
function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcamElement, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.7);
}

// Update the gaze dot position
function updateGazeDot(x, y) {
    gazeDot.style.left = `${x}px`;
    gazeDot.style.top = `${y}px`;
}

// Update the heatmap with gaze data
function updateHeatmap() {
    // Keep only the most recent 100 points
    if (gazeData.length > 100) {
        gazeData = gazeData.slice(-100);
    }
    
    // Map webcam coordinates to marketing image coordinates
    const webcamRect = webcamElement.getBoundingClientRect();
    const marketingContent = document.getElementById('marketingContent');
    const marketingRect = marketingContent.getBoundingClientRect();
    
    // Clear existing points
    heatmapInstance.clear();
    
    // Convert gaze data to heatmap points
    const points = gazeData.map(data => {
        // Convert gaze x,y from webcam to relative values (0-1)
        const relX = (data.x - webcamRect.left) / webcamRect.width;
        const relY = (data.y - webcamRect.top) / webcamRect.height;
        
        // Apply to marketing image dimensions
        return {
            x: Math.floor(relX * marketingRect.width),
            y: Math.floor(relY * marketingRect.height),
            value: 1
        };
    });
    
    // Add points to heatmap
    heatmapInstance.addData(points);
}

// Create emotion bar UI element
function createEmotionBar(emotion, value) {
    const container = document.createElement('div');
    container.className = 'emotion-bar-container';
    
    const label = document.createElement('div');
    label.className = 'emotion-label';
    label.textContent = emotion;
    
    const barOuter = document.createElement('div');
    barOuter.className = 'emotion-bar-outer';
    
    const barInner = document.createElement('div');
    barInner.className = 'emotion-bar-inner';
    barInner.style.width = `${value * 100}%`;
    barInner.style.backgroundColor = emotionColors[emotion];
    
    const valueLabel = document.createElement('div');
    valueLabel.className = 'emotion-value';
    valueLabel.textContent = `${Math.round(value * 100)}%`;
    
    barOuter.appendChild(barInner);
    container.appendChild(label);
    container.appendChild(barOuter);
    container.appendChild(valueLabel);
    
    emotionBars.appendChild(container);
}

// Update emotion display with new prediction results
function updateEmotionDisplay(result) {
    // Update emotion label
    emotionLabel.textContent = result.top_emotion;
    emotionLabel.style.color = emotionColors[result.top_emotion];
    
    // Update emotion bars
    const barContainers = emotionBars.querySelectorAll('.emotion-bar-container');
    
    barContainers.forEach((container, i) => {
        const emotion = Object.keys(emotionColors)[i];
        const value = result.emotions[emotion];
        
        const barInner = container.querySelector('.emotion-bar-inner');
        const valueLabel = container.querySelector('.emotion-value');
        
        barInner.style.width = `${value * 100}%`;
        valueLabel.textContent = `${Math.round(value * 100)}%`;
    });
}

// Update engagement metrics display
function updateEngagementMetrics(result) {
    if (result.engagement_metrics) {
        engagementScore.textContent = `${result.engagement_metrics.score}%`;
        attentionDuration.textContent = result.engagement_metrics.attention_duration;
        emotionalImpact.textContent = `${result.engagement_metrics.emotional_impact}%`;
    }
}

// Update time series chart with new emotion data
function updateTimeSeriesChart(result) {
    // Add data to chart
    const timestamp = new Date().toLocaleTimeString();
    
    // Add timestamp for x-axis
    timeSeriesChart.data.labels.push(timestamp);
    
    // Limit x-axis to 30 data points
    if (timeSeriesChart.data.labels.length > 30) {
        timeSeriesChart.data.labels.shift();
    }
    
    // Add data for each emotion
    timeSeriesChart.data.datasets.forEach((dataset, i) => {
        const emotion = dataset.label;
        dataset.data.push(result.emotions[emotion]);
        
        // Limit data points
        if (dataset.data.length > 30) {
            dataset.data.shift();
        }
    });
    
    // Update chart
    timeSeriesChart.update();
}
