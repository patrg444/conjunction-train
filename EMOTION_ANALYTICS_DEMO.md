# EmotionTrack: Marketing Analytics Platform

## Overview

EmotionTrack is a cutting-edge marketing analytics platform that combines real-time emotion recognition with eye tracking to deliver unprecedented insights into customer reactions. This demo showcases the power of our multimodal emotion recognition model with synchronized data augmentation, offering a glimpse into the future of marketing analytics.

## Key Features

- **Real-time Emotion Recognition**: Analyze facial expressions and voice to detect 6 core emotions
- **Eye Tracking Integration**: Track where users look and correlate with emotional responses
- **Attention Heatmaps**: Visualize which parts of marketing content generate the most attention
- **Engagement Metrics**: Quantify emotional impact and attention duration
- **Temporal Analysis**: Track emotional responses over time

## Technical Highlights

- LSTM-based deep learning architecture with attention mechanisms
- Synchronized multi-modal processing of audio and video streams
- 15 FPS processing for smooth real-time performance
- FaceNet embeddings for robust facial feature extraction
- WebGazer.js integration for browser-based eye tracking
- Flask backend for easy deployment and scaling

## Getting Started

### Prerequisites

- Python 3.7+ with TensorFlow 2.x
- Webcam and microphone
- Modern web browser with JavaScript enabled

### Installation

1. Clone this repository
2. Ensure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```
   ./run_emotion_demo.sh
   ```
4. Open your browser to http://localhost:5000

### Using the Demo

1. **Calibrate the Eye Tracking**:
   - Click the "Calibrate Eye Tracking" button
   - Follow the on-screen instructions to look at and click each calibration point
   - Finish calibration when complete

2. **Run the Analysis**:
   - Click "Start Analysis" to begin real-time emotion recognition and eye tracking
   - Interact naturally with the webcam (expressions and speech)
   - The system will display your current emotional state and eye tracking data
   - View the heatmap to see where your attention is focused
   - Check the emotion timeline for changes in emotional states over time

3. **Stop the Analysis**:
   - Click "Stop" to end the session

## Investor Demonstration Guide

For the investor presentation, demonstrate these key capabilities:

1. **Technical Robustness**:
   - Show the real-time emotion detection with various facial expressions
   - Demonstrate how the system maintains accuracy despite movement and lighting changes

2. **Eye Tracking Correlation**:
   - Display how the system tracks gaze in real-time
   - Show the heatmap developing as attention is given to different parts of the marketing content

3. **Business Applications**:
   - Explain how the engagement metrics translate to ROI for marketing teams
   - Demonstrate how emotional responses are quantified and visualized for business insights

4. **Market Potential**:
   - Highlight the competitive advantages over existing solutions
   - Discuss scaling opportunities for enterprise clients

## Business Model

EmotionTrack offers a tiered SaaS pricing model:

1. **Professional**: $499/month
   - For small agencies and individual marketers
   - Basic emotion tracking and eye tracking
   - Limited sessions per month

2. **Business**: $1,999/month
   - For mid-sized agencies and brands
   - Advanced analytics and reporting
   - Integration with marketing platforms
   - Unlimited sessions

3. **Enterprise**: $7,999+/month
   - For major brands and agencies
   - Custom integration and deployment
   - White-labeling options
   - Dedicated support and customization

## Technical Architecture

```
┌────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Web Frontend │     │  Flask Backend  │     │  LSTM Attention │
│  (HTML/JS/CSS) │     │   (Python)      │     │    Model        │
│                │     │                 │     │                 │
│  - WebGazer.js │────►│  - Face/Audio   │────►│  - Multimodal   │
│  - UI/Controls │     │    Processing   │     │    Processing   │
│  - Visualization│◄───│  - API Endpoints│◄────│  - Emotion      │
└────────────────┘     └─────────────────┘     │    Classification│
                                               └─────────────────┘
```

## Future Development

1. **Expanded Emotions**: Extend to more complex emotional states
2. **Mobile Integration**: Develop iOS and Android SDKs
3. **API Ecosystem**: Create developer tools for third-party integration
4. **Personalization Engine**: Adapt content based on emotional responses

## Contact

For more information, please contact:
- Email: info@emotiontrack.ai
- Website: www.emotiontrack.ai
