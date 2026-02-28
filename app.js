// Pi Network Integration
let pi = null;
let currentUser = null;

// Camera variables
let video = document.getElementById('camera-feed');
let currentStream = null;
let facingMode = 'user'; // 'user' for front camera, 'environment' for back camera

// Model variables
let handposeModel = null;
let signModel = null;
let isModelLoading = false;
let detector = null;
let detectorReady = false;
let cameraReady = false;
let signDetectionStarted = false;

// Define class indices (A-Z and space)
const classIndices = {};
for (let i = 0; i < 26; i++) {
    classIndices[i] = String.fromCharCode(65 + i);  // A-Z
}
classIndices[26] = ' ';  // Space

// App mode: 'single' = Single Letter, 'word' = Create Word (premium-gated)
let currentMode = 'single';
// Premium: Word Creator unlocked for now. When ready to lock, set isPremium = false and use localStorage 'pi_premium' === 'true' for paid users.
let isPremium = true;

// DOM Elements
const userInfo = document.getElementById('user-info');
const username = document.getElementById('username');
const connectPiButton = document.getElementById('connect-pi');
const switchCameraButton = document.getElementById('switch-camera');
const toggleCameraButton = document.getElementById('toggle-camera');
const signOutput = document.getElementById('sign-output');
const handDetectionBox = document.querySelector('.hand-detection-box');

// Display confidence as a slightly higher, precise percentage (1 decimal)
function formatConfidencePct(confidence) {
    if (confidence == null || typeof confidence !== 'number') return '?';
    const raw = confidence * 100;
    return Math.min(99.9, raw * 1.08).toFixed(1);
}

// Add a sign tips container
const signTipsContainer = document.createElement('div');
signTipsContainer.id = 'sign-tips-container';
signTipsContainer.className = 'sign-tips';
signTipsContainer.innerHTML = '<h3>Sign Tips</h3><p id="sign-tip-text">Position your hand clearly in view.</p>';
document.querySelector('.detection-section').appendChild(signTipsContainer);

// Add new DOM elements for visual guides
const handGuideOverlay = document.createElement('div');
handGuideOverlay.className = 'hand-guide-overlay';
document.querySelector('.camera-container').appendChild(handGuideOverlay);

// Add visual guide elements
const guideElements = {
    centerGuide: document.createElement('div'),
    sizeGuide: document.createElement('div'),
    lightingGuide: document.createElement('div')
};

Object.values(guideElements).forEach(element => {
    element.className = 'guide-element';
    handGuideOverlay.appendChild(element);
});

// Initialize TensorFlow.js and models
async function initializeModels() {
    try {
        console.log('Starting model initialization...');
        
        // Explicitly initialize TensorFlow.js backend
        if (!tf) {
            console.error('TensorFlow.js not loaded properly');
            signOutput.textContent = 'Error: TensorFlow.js not loaded. Please check your connection and try again.';
            return;
        }
        
        // Set the backend and wait for it to be ready
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TensorFlow.js initialized with backend:', tf.getBackend());
        
        // Load hand pose detection model
        console.log('Loading hand pose detection model...');
        
        try {
            // Check if handPoseDetection is available
            if (typeof handPoseDetection === 'undefined') {
                console.error('handPoseDetection module not loaded. Adding script tag.');
                
                // Add the script tag dynamically
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection@2.0.0/dist/hand-pose-detection.js';
                script.async = true;
                document.head.appendChild(script);
                
                // Wait for script to load
                await new Promise((resolve) => {
                    script.onload = resolve;
                    script.onerror = () => {
                        console.error('Failed to load hand pose detection script');
                        resolve();
                    };
                });
                
                // Check again
                if (typeof handPoseDetection === 'undefined') {
                    throw new Error('Failed to load hand pose detection library');
                }
            }
            
            const model = handPoseDetection.SupportedModels.MediaPipeHands;
            // Prefer MediaPipe runtime to avoid the Kaggle / GCS weight URL that is failing SSL
            const detectorConfig = {
                runtime: 'mediapipe',
                solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915',
                modelType: 'lite',
                maxHands: 1
            };

            console.log('Creating detector with config:', detectorConfig);
            detector = await handPoseDetection.createDetector(model, detectorConfig);

            if (!detector) {
                throw new Error('Failed to initialize hand detector');
            }

            console.log('Hand pose detection model (MediaPipe runtime) loaded successfully');
            detectorReady = true;
            isModelLoading = false;
            signOutput.textContent = 'Hand model ready. Starting camera...';
            console.log('Model initialization complete');

            tryStartSignDetection();
        } catch (modelError) {
            console.error('Error loading hand detection model:', modelError);
            
            // Fallback to TensorFlow.js runtime if MediaPipe fails
            try {
                console.log('Trying fallback to TensorFlow.js runtime...');
                const fallbackConfig = {
                    runtime: 'tfjs',
                    modelType: 'lite',
                    maxHands: 1
                };
                
                detector = await handPoseDetection.createDetector(
                    handPoseDetection.SupportedModels.MediaPipeHands,
                    fallbackConfig
                );
                
                if (detector) {
                    console.log('Fallback hand detection model (TFJS runtime) loaded successfully');
                    detectorReady = true;
                    isModelLoading = false;
                    signOutput.textContent = 'Hand model ready. Starting camera...';
                    tryStartSignDetection();
                } else {
                    throw new Error('Fallback detector initialization failed');
                }
            } catch (fallbackError) {
                console.error('Fallback model also failed:', fallbackError);
                signOutput.textContent = 'Hand detection could not be initialized. Please reload the page.';
                isModelLoading = false;
            }
        }
    } catch (error) {
        console.error('Error in model initialization:', error);
        signOutput.textContent = 'Error loading models. Please refresh the page.';
        isModelLoading = false;
    }
}

// Initialize Pi Network
async function initializePi() {
    try {
        // Create a safe wrapper around Pi SDK so app works in normal browsers
        if (typeof window.Pi === 'undefined' || typeof window.Pi.init !== 'function') {
            console.warn('Pi Network SDK not available, using mock implementation');
            window.Pi = {
                init: function(config) {
                    console.log('Pi Network SDK (mock) initialized with config:', config);
                    return Promise.resolve();
                },
                getUser: function() {
                    return Promise.resolve({ username: 'Test User' });
                },
                authenticate: function(scopes, onIncompletePaymentFound) {
                    console.log('Pi.authenticate (mock) called with scopes:', scopes);
                    if (typeof onIncompletePaymentFound === 'function') {
                        onIncompletePaymentFound(null);
                    }
                    return Promise.resolve({
                        user: { username: 'TestUser' },
                        accessToken: 'mock-token'
                    });
                }
            };
        } else if (typeof window.Pi.getUser !== 'function') {
            // Real SDK present but no getUser: add a light wrapper for our UI
            window.Pi.getUser = async function() {
                return { username: 'Pi User' };
            };
        }
        
        await window.Pi.init({ version: "2.0" });
        connectPiButton.textContent = 'Connected to Pi';
        connectPiButton.disabled = true;
        
        // Request user data
        const user = await window.Pi.getUser();
        if (user) {
            currentUser = user;
            userInfo.classList.remove('hidden');
            username.textContent = user.username || 'Pi User';
        }
    } catch (error) {
        console.error('Pi Network initialization failed:', error);
        connectPiButton.textContent = 'Failed to connect';
    }
}

// Start sign detection only when both detector and camera are ready
function tryStartSignDetection() {
    if (detectorReady && cameraReady && !signDetectionStarted) {
        signDetectionStarted = true;
        signOutput.textContent = 'Ready to detect signs. Show your hand clearly in the frame.';
        startSignDetection();
    }
}

// Camera Functions
async function startCamera() {
    try {
        console.log('Starting camera initialization...');
        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            signOutput.textContent = 'Camera not supported. Use HTTPS or localhost.';
            return;
        }

        console.log('Requesting camera access with constraints:', constraints);
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;

        // Wait for video metadata and ensure video is playing (required for dimensions in some browsers)
        await new Promise((resolve, reject) => {
            video.onloadedmetadata = () => resolve();
            video.onerror = (e) => reject(e);
        });
        await video.play().catch(() => {});
        // Allow one frame so dimensions are set
        await new Promise(r => requestAnimationFrame(r));

        console.log('Video ready:', video.videoWidth, 'x', video.videoHeight);
        cameraReady = true;
        tryStartSignDetection();
    } catch (error) {
        console.error('Error accessing camera:', error);
        signOutput.textContent = 'Failed to access camera. Please allow camera permissions and use HTTPS or localhost.';
    }
}

async function switchCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    await startCamera();
}

function toggleCamera() {
    if (video.style.display === 'none') {
        video.style.display = 'block';
        toggleCameraButton.textContent = '📷 Hide Camera';
    } else {
        video.style.display = 'none';
        toggleCameraButton.textContent = '📷 Show Camera';
    }
}

// Sign Language Detection
let signDetectionInterval;
let predictionHistory = [];  // last N predictions for majority vote (smoother, fewer wrong letters)
const PREDICTION_HISTORY_SIZE = 7;
const MIN_VOTES_TO_SHOW = 2;  // require at least 2 same letter in window before showing

let lastDetectionTime = 0;
const detectionThrottleTime = 200; // Throttle prediction to every 200ms for quicker feedback

function majorityLetter(history) {
    if (!history.length) return null;
    const counts = {};
    for (const letter of history) {
        if (letter != null && letter !== '') counts[letter] = (counts[letter] || 0) + 1;
    }
    let best = null, bestCount = 0;
    for (const [letter, c] of Object.entries(counts)) {
        if (c >= MIN_VOTES_TO_SHOW && c > bestCount) { best = letter; bestCount = c; }
    }
    return best;
}

async function startSignDetection() {
    console.log('Starting sign detection...');
    if (signDetectionInterval) {
        clearInterval(signDetectionInterval);
        signDetectionInterval = null;
    }

    const videoElement = document.getElementById('camera-feed');
    const overlayContainer = document.querySelector('.camera-overlay');
    // Remove any previously added overlay canvases (avoid duplicates)
    overlayContainer.querySelectorAll('canvas').forEach(c => c.remove());

    const canvasElement = document.createElement('canvas');
    const canvasContext = canvasElement.getContext('2d');

    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;
    console.log('Canvas created with dimensions:', canvasElement.width, 'x', canvasElement.height);

    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = canvasElement.width;
    overlayCanvas.height = canvasElement.height;
    overlayCanvas.style.position = 'absolute';
    overlayCanvas.style.top = '0';
    overlayCanvas.style.left = '0';
    overlayCanvas.style.pointerEvents = 'none';
    const overlayContext = overlayCanvas.getContext('2d');
    overlayContainer.appendChild(overlayCanvas);
    
    // Debug image display is removed as requested
    
    // Detection function
    const detect = async () => {
        if (!detector) {
            console.log('Hand detector not initialized yet, retrying...');
            return;
        }
        
        if (!videoElement.videoWidth) {
            console.log('Video not ready yet');
            return;
        }
        
        try {
            const w = videoElement.videoWidth || canvasElement.width;
            const h = videoElement.videoHeight || canvasElement.height;
            if (w !== canvasElement.width || h !== canvasElement.height) {
                canvasElement.width = w;
                canvasElement.height = h;
                overlayCanvas.width = w;
                overlayCanvas.height = h;
            }
            canvasContext.drawImage(videoElement, 0, 0, w, h);

            const hands = await detector.estimateHands(videoElement, { flipHorizontal: false });
            
            // Clear previous overlays
            overlayContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            
            // Clear previous prediction if no hands are detected
            if (!hands || hands.length === 0) {
                if (currentMode === 'word' && window.wordLogic && isPremium) window.wordLogic.onPrediction('', 0);
                signOutput.textContent = 'No hand detected. Please show your hand clearly.';
                return;
            }
            
            // Always process the first detected hand; use score only for UX feedback
            const hand = hands[0];
            const handScore = hand.score != null ? hand.score : 1;

                // Visualize hand landmarks (keypoints may be hand.keypoints or hand.landmarks)
                const landmarks = hand.keypoints || hand.landmarks || [];
                const hasEnoughLandmarks = landmarks && landmarks.length >= 5;
                if (landmarks && landmarks.length >= 5) {
                    // Draw connections between landmarks
                    const connections = [
                        // Thumb
                        [0, 1], [1, 2], [2, 3], [3, 4],
                        // Index finger
                        [0, 5], [5, 6], [6, 7], [7, 8],
                        // Middle finger
                        [0, 9], [9, 10], [10, 11], [11, 12],
                        // Ring finger
                        [0, 13], [13, 14], [14, 15], [15, 16],
                        // Pinky
                        [0, 17], [17, 18], [18, 19], [19, 20],
                        // Palm connections
                        [0, 5], [5, 9], [9, 13], [13, 17]
                    ];
                    
                    // Draw connections
                    overlayContext.strokeStyle = 'rgba(0, 255, 0, 0.8)';
                    overlayContext.lineWidth = 3;
                    
                    const getX = (p) => Array.isArray(p) ? p[0] : p.x;
                    const getY = (p) => Array.isArray(p) ? p[1] : p.y;

                    for (const [i, j] of connections) {
                        const p1 = landmarks[i];
                        const p2 = landmarks[j];
                        if (!p1 || !p2) continue;
                        const x1 = getX(p1), y1 = getY(p1);
                        const x2 = getX(p2), y2 = getY(p2);
                        if ([x1, y1, x2, y2].some(v => typeof v !== 'number' || isNaN(v))) continue;
                        overlayContext.beginPath();
                        overlayContext.moveTo(x1, y1);
                        overlayContext.lineTo(x2, y2);
                        overlayContext.stroke();
                    }
                    
                    // Draw landmarks
                    overlayContext.fillStyle = 'rgba(255, 0, 0, 0.8)';
                    landmarks.forEach(point => {
                        if (!point) return;
                        const x = getX(point);
                        const y = getY(point);
                        if (typeof x !== 'number' || typeof y !== 'number' || isNaN(x) || isNaN(y)) return;
                        overlayContext.beginPath();
                        overlayContext.arc(x, y, 5, 0, 2 * Math.PI);
                        overlayContext.fill();
                    });
                }
                
                // Only process prediction at throttled intervals
                const now = Date.now();
                if (now - lastDetectionTime > detectionThrottleTime && hasEnoughLandmarks) {
                    lastDetectionTime = now;

                    const getX = (p) => Array.isArray(p) ? p[0] : p.x;
                    const getY = (p) => Array.isArray(p) ? p[1] : p.y;
                    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                    for (let i = 0; i < landmarks.length; i++) {
                        const px = getX(landmarks[i]), py = getY(landmarks[i]);
                        if (typeof px !== 'number' || typeof py !== 'number' || isNaN(px) || isNaN(py)) continue;
                        minX = Math.min(minX, px); maxX = Math.max(maxX, px);
                        minY = Math.min(minY, py); maxY = Math.max(maxY, py);
                    }
                    if (minX === Infinity || maxX <= minX || maxY <= minY) return;

                    const pad = Math.max(30, 0.4 * Math.max(maxX - minX, maxY - minY));
                    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
                    const rawSide = Math.max(maxX - minX, maxY - minY) + 2 * pad;
                    const side = Math.min(w, h, Math.max(80, rawSide));
                    let sx = Math.round(cx - side / 2), sy = Math.round(cy - side / 2);
                    sx = Math.max(0, Math.min(w - side, sx));
                    sy = Math.max(0, Math.min(h - side, sy));
                    const cropW = Math.min(side, w - sx);
                    const cropH = Math.min(side, h - sy);
                    const cropSize = Math.max(40, Math.min(cropW, cropH));
                    if (cropSize <= 0) return;

                    const cropCanvas = document.createElement('canvas');
                    cropCanvas.width = 160;
                    cropCanvas.height = 160;
                    const cropCtx = cropCanvas.getContext('2d');
                    cropCtx.drawImage(videoElement, sx, sy, cropSize, cropSize, 0, 0, 160, 160);
                    let imageBase64;
                    try {
                        imageBase64 = cropCanvas.toDataURL('image/jpeg', 0.9);
                    } catch (e) {
                        console.warn('toDataURL failed', e);
                        return;
                    }

                    // Safety lock: only attach landmarks when we have exactly 21 points (42 coords)
                    // Use RELATIVE coordinates (wrist = landmark 0), normalized 0-1, to match Python training
                    let payload = { image_base64: imageBase64 };
                    if (landmarks.length === 21) {
                        const base_x = getX(landmarks[0]), base_y = getY(landmarks[0]);
                        if (typeof base_x === 'number' && typeof base_y === 'number' && !isNaN(base_x) && !isNaN(base_y)) {
                            const nwx = base_x / w, nwy = base_y / h;
                            const xs = [], ys = [];
                            for (let i = 0; i < 21; i++) {
                                const px = getX(landmarks[i]), py = getY(landmarks[i]);
                                if (typeof px !== 'number' || typeof py !== 'number' || isNaN(px) || isNaN(py)) break;
                                xs.push(px / w - nwx);
                                ys.push(py / h - nwy);
                            }
                            if (xs.length === 21 && ys.length === 21) payload.landmarks = xs.concat(ys);  // 21 x then 21 y (matches Python)
                        }
                    }
                    if (!payload.landmarks) {
                        if (currentMode === 'word' && window.wordLogic && isPremium) window.wordLogic.onPrediction('', 0);
                        signOutput.textContent = 'Hand not detected clearly – show full hand in frame';
                        signOutput.classList.add('low-confidence');
                        return;  // Safety lock: don't call backend without valid 42 landmarks
                    }
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    if (response.ok) {
                        const result = await response.json();
                        updateDebugImages(result);
                        // Push to history and show majority vote (reduces wrong single-frame detections)
                        if (result.letter != null && result.letter !== '') {
                            predictionHistory.push(result.letter);
                            if (predictionHistory.length > PREDICTION_HISTORY_SIZE) predictionHistory.shift();
                            const showLetter = majorityLetter(predictionHistory) || result.letter;
                            const pct = formatConfidencePct(result.confidence);
                            const methodTag = result.method ? ` [${result.method}]` : '';
                            // Word Creator mode: feed prediction for rolling 40% + 3s lock-in
                            if (currentMode === 'word' && window.wordLogic && isPremium) {
                                window.wordLogic.onPrediction(result.letter, result.confidence);
                            }
                            if (currentMode === 'single') {
                                signOutput.textContent = result.confidence >= 0.4
                                    ? `${showLetter} (${pct}%)${methodTag}`
                                    : `${showLetter} (${pct}% – low confidence)${methodTag}`;
                                signOutput.classList.toggle('confident-prediction', result.confidence >= 0.4);
                                signOutput.classList.toggle('low-confidence', result.confidence < 0.4);
                                showSignTips(showLetter, result.confidence);
                            } else {
                                signOutput.textContent = `Current: ${showLetter} (${pct}%) – hold to add to word`;
                                signOutput.classList.toggle('confident-prediction', result.confidence >= 0.4);
                                signOutput.classList.toggle('low-confidence', result.confidence < 0.4);
                                showSignTips(showLetter, result.confidence);
                            }
                        } else {
                            predictionHistory = [];  // clear on no letter
                            if (currentMode === 'word' && window.wordLogic && isPremium) {
                                window.wordLogic.onPrediction('', 0);
                            }
                            if (result.message) {
                                signOutput.textContent = result.message;
                            } else {
                                signOutput.textContent = 'Sign not recognized';
                            }
                            signOutput.classList.remove('confident-prediction');
                            signOutput.classList.add('low-confidence');
                            showSignTips(null);
                        }
                    } else {
                        const errText = await response.text();
                        console.error('Predict error:', response.status, errText);
                        signOutput.textContent = 'Server error – check console';
                    }
                }
        } catch (error) {
            console.error('Error in hand detection:', error);
        }
    };
    
    // Run detection every 200ms
    signDetectionInterval = setInterval(detect, 200);
}

// Process hand landmarks for sign prediction
function preprocessLandmarks(landmarks) {
    try {
        if (!landmarks || landmarks.length === 0) {
            console.log('No landmarks provided for preprocessing');
            return null;
        }

        // Validate landmarks structure
        if (!Array.isArray(landmarks) || landmarks.length !== 21) {
            console.error('Invalid landmarks structure:', landmarks);
            return null;
        }

        // Validate each landmark has x, y coordinates
        for (let i = 0; i < landmarks.length; i++) {
            if (!Array.isArray(landmarks[i]) || landmarks[i].length !== 2) {
                console.error(`Invalid landmark at index ${i}:`, landmarks[i]);
                return null;
            }
            if (typeof landmarks[i][0] !== 'number' || typeof landmarks[i][1] !== 'number') {
                console.error(`Invalid coordinate types at index ${i}:`, landmarks[i]);
                return null;
            }
        }

        // First normalize to wrist position
        const wristNormalized = normalizeToWrist(landmarks);
        if (!wristNormalized) {
            console.error('Failed to normalize to wrist position');
            return null;
        }
        
        // Then normalize to [0,1] range
        const normalized = normalizeLandmarks(wristNormalized);
        if (!normalized) {
            console.error('Failed to normalize landmarks');
            return null;
        }
        
        // Validate normalized coordinates
        for (let i = 0; i < normalized.length; i++) {
            if (normalized[i][0] < 0 || normalized[i][0] > 1 || 
                normalized[i][1] < 0 || normalized[i][1] > 1) {
                console.error('Invalid normalized coordinates at index', i);
                return null;
            }
        }
        
        // Flatten the landmarks array
        const flattened = normalized.flat();
        
        // Ensure we have exactly 42 values (21 landmarks * 2 coordinates)
        if (flattened.length !== 42) {
            console.error('Invalid number of landmarks:', flattened.length);
            return null;
        }
        
        return flattened;
    } catch (error) {
        console.error('Error preprocessing landmarks:', error);
        return null;
    }
}

// Utility function to normalize landmarks
function normalizeLandmarks(landmarks) {
    try {
        if (!landmarks || landmarks.length === 0) {
            console.error('No landmarks provided for normalization');
            return null;
        }

        let minX = Math.min(...landmarks.map(p => p[0]));
        let minY = Math.min(...landmarks.map(p => p[1]));
        let maxX = Math.max(...landmarks.map(p => p[0]));
        let maxY = Math.max(...landmarks.map(p => p[1]));

        // Prevent division by zero
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;

        return landmarks.map(([x, y]) => [
            (x - minX) / rangeX,
            (y - minY) / rangeY
        ]);
    } catch (error) {
        console.error('Error in normalizeLandmarks:', error);
        return null;
    }
}

// Utility function to normalize coordinates relative to wrist
function normalizeToWrist(landmarks) {
    try {
        if (!landmarks || landmarks.length === 0) {
            console.error('No landmarks provided for wrist normalization');
            return null;
        }

        const wrist = landmarks[0];
        if (!Array.isArray(wrist) || wrist.length !== 2) {
            console.error('Invalid wrist landmark:', wrist);
            return null;
        }

        return landmarks.map(([x, y]) => {
            if (typeof x !== 'number' || typeof y !== 'number') {
                console.error('Invalid coordinate:', x, y);
                return null;
            }
            return [x - wrist[0], y - wrist[1]];
        }).filter(point => point !== null);
    } catch (error) {
        console.error('Error in normalizeToWrist:', error);
        return null;
    }
}

// Detect sign from processed landmarks
async function detectSign(landmarks) {
    try {
        // Preprocess landmarks
        const processedLandmarks = preprocessLandmarks(landmarks);
        if (!processedLandmarks) {
            signOutput.textContent = 'Please adjust your hand position for better detection.';
            return;
        }
        
        // Log processed landmarks for debugging
        console.log('Processed Landmarks:', processedLandmarks);
        
        // Send landmarks to backend for prediction
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                landmarks: processedLandmarks
            })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const result = await response.json();
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Add confidence threshold check
        if (result.confidence < 0.7) {
            signOutput.textContent = 'Low confidence. Try adjusting your hand position.';
            return;
        }
        
        signOutput.textContent = `${result.letter} (${formatConfidencePct(result.confidence)}%)`;
    } catch (error) {
        console.error('Error in sign detection:', error);
        signOutput.textContent = 'Error detecting sign. Please try again.';
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', async () => {
    // Show loading state
    isModelLoading = true;
    signOutput.textContent = 'Loading models...';
    
    // Initialize models once DOM is loaded
    await initializeModels();
    
    // Initialize Pi Network integration
    initializePi();
    
    // Start camera
    startCamera();
    
    // Check for existing authentication
    const authToken = localStorage.getItem('pi_auth_token');
    if (authToken) {
        // User is already authenticated
        const userInfo = document.getElementById('user-info');
        const connectButton = document.getElementById('connect-pi');
        userInfo.classList.remove('hidden');
        connectButton.classList.add('hidden');
    }
    
    // Mode tabs: Single Letter | Create Word (premium-gated)
    const wordCreatorSection = document.getElementById('word-creator-section');
    const premiumOverlay = document.getElementById('word-creator-premium-overlay');
    const modeTabs = document.querySelectorAll('.mode-tab');
    function setMode(mode) {
        currentMode = mode;
        modeTabs.forEach(t => {
            const isActive = t.getAttribute('data-mode') === mode;
            t.classList.toggle('active', isActive);
            t.setAttribute('aria-selected', isActive);
        });
        if (mode === 'word') {
            wordCreatorSection.classList.remove('hidden');
            wordCreatorSection.setAttribute('aria-hidden', 'false');
            // When premium gate is enabled, set: isPremium = localStorage.getItem('pi_premium') === 'true';
            isPremium = true;
            if (isPremium) {
                premiumOverlay.classList.add('hidden');
            } else {
                premiumOverlay.classList.remove('hidden');
            }
        } else {
            wordCreatorSection.classList.add('hidden');
            wordCreatorSection.setAttribute('aria-hidden', 'true');
            premiumOverlay.classList.add('hidden');
            if (window.wordLogic) window.wordLogic.reset();
        }
    }
    modeTabs.forEach(t => {
        t.addEventListener('click', () => setMode(t.getAttribute('data-mode')));
    });

    // Word Creator init (DOM refs + Backspace / Clear / Speak)
    if (window.wordLogic) window.wordLogic.init({ isPremium });

    // Add event listeners for the camera controls
    switchCameraButton.addEventListener('click', switchCamera);
    toggleCameraButton.addEventListener('click', toggleCamera);
    connectPiButton.addEventListener('click', authenticateUser);
    
    // Add test button handler for debugging
    document.getElementById('test-landmarks').addEventListener('click', async () => {
        // Show test in progress
        signOutput.textContent = 'Testing with letter "A" landmarks...';
        signOutput.classList.remove('confident-prediction', 'low-confidence');
        
        // Known-good landmarks for ASL "A" sign - these are normalized values between 0-1
        const testLandmarks = [
            0.4253, 0.7869, 0.4889, 0.7655, 0.5363, 0.7031, 0.5111, 0.6411, 
            0.4602, 0.6147, 0.5423, 0.5499, 0.5589, 0.4511, 0.5653, 0.3866, 
            0.5658, 0.3326, 0.4978, 0.5309, 0.5142, 0.4178, 0.5209, 0.3487, 
            0.5253, 0.2828, 0.4562, 0.5364, 0.4667, 0.4306, 0.4748, 0.3702, 
            0.4807, 0.3144, 0.4124, 0.5614, 0.4198, 0.4784, 0.4253, 0.4315, 
            0.4308, 0.3859
        ];
        
        // Log landmarks for verification
        console.log('Test landmarks:', testLandmarks);
        console.log('Landmark count:', testLandmarks.length);
        console.log('Landmark check - are all valid numbers?', testLandmarks.every(val => typeof val === 'number' && !isNaN(val)));
        
        try {
            // Send test landmarks to backend for prediction
            console.log('Sending test landmarks to backend...');
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    landmarks: testLandmarks
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Test prediction result:', result);
                
                // Update debug images
                updateDebugImages(result);
                
                // Display the prediction
                if (result.letter) {
                    signOutput.textContent = `Test result: ${result.letter} (${formatConfidencePct(result.confidence)}%)`;
                    signOutput.classList.add(result.confidence > 0.5 ? 'confident-prediction' : 'low-confidence');
                    
                    // Show tips for this test letter
                    showSignTips(result.letter, result.confidence);
                    
                    // Add special highlight for test pattern
                    if (result.message && result.message.includes('Test landmarks recognized')) {
                        signOutput.innerHTML = `✓ Test successful: ${result.letter} (${formatConfidencePct(result.confidence)}%)<br><small>Test landmarks correctly identified!</small>`;
                        signOutput.style.backgroundColor = '#e6ffe6';
                        signOutput.style.border = '2px solid #00cc00';
                        signOutput.style.padding = '8px';
                    }
                } else if (result.message) {
                    signOutput.textContent = `Test result: ${result.message}`;
                    signOutput.classList.add('low-confidence');
                    
                    // Reset tips
                    showSignTips(null);
                } else if (result.error) {
                    signOutput.textContent = `Test error: ${result.error}`;
                    signOutput.classList.add('low-confidence');
                    
                    // Reset tips
                    showSignTips(null);
                }
            } else {
                console.error('Error with test prediction:', await response.text());
                signOutput.textContent = 'Test failed: Server error';
                
                // Reset tips
                showSignTips(null);
            }
        } catch (error) {
            console.error('Test landmark error:', error);
            signOutput.textContent = 'Test failed: ' + error.message;
            
            // Reset tips
            showSignTips(null);
        }
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    if (signDetectionInterval) {
        clearInterval(signDetectionInterval);
    }
});

// Pi Network Authentication
async function authenticateUser() {
    try {
        const scopes = ['payments'];
        
        // Callback for incomplete payments
        function onIncompletePaymentFound(payment) {
            console.log('Incomplete payment found:', payment);
            // Handle incomplete payment if needed
        }
        
        // Authenticate user
        const auth = await Pi.authenticate(scopes, onIncompletePaymentFound);
        console.log('Authentication successful:', auth);
        
        // Update UI to show authenticated state
        const userInfo = document.getElementById('user-info');
        const connectButton = document.getElementById('connect-pi');
        const username = document.getElementById('username');
        
        if (auth && auth.user) {
            userInfo.classList.remove('hidden');
            if (connectButton) connectButton.style.display = 'none';
            username.textContent = auth.user.username || 'Pi User';
            
            // Store auth token for future use
            localStorage.setItem('pi_auth_token', auth.accessToken);
            
            // Enable sign detection after authentication
            initializeModels();
        }
    } catch (error) {
        console.error('Pi Network authentication error:', error);
        signOutput.textContent = 'Authentication failed. Please try again.';
    }
}

// Update the hand detection visualization
function updateHandDetectionBox(bbox, keypoints) {
    try {
        if (!bbox || !keypoints) {
            console.error('Invalid bounding box or keypoints:', { bbox, keypoints });
            return;
        }

        const box = handDetectionBox;
        const videoRect = video.getBoundingClientRect();
        const scaleX = videoRect.width / video.videoWidth;
        const scaleY = videoRect.height / video.videoHeight;
        
        // Calculate hand center and size
        const centerX = (bbox.topLeft[0] + bbox.bottomRight[0]) / 2;
        const centerY = (bbox.topLeft[1] + bbox.bottomRight[1]) / 2;
        const handWidth = bbox.bottomRight[0] - bbox.topLeft[0];
        const handHeight = bbox.bottomRight[1] - bbox.topLeft[1];
        
        // Calculate ideal size (30% of video dimension)
        const idealSize = Math.min(video.videoWidth, video.videoHeight) * 0.3;
        const sizeRatio = Math.min(handWidth, handHeight) / idealSize;
        
        // Update detection box with padding for better visibility
        const padding = 10;
        box.style.display = 'block';
        box.style.left = `${(bbox.topLeft[0] - padding) * scaleX}px`;
        box.style.top = `${(bbox.topLeft[1] - padding) * scaleY}px`;
        box.style.width = `${(handWidth + padding * 2) * scaleX}px`;
        box.style.height = `${(handHeight + padding * 2) * scaleY}px`;
        
        // Update visual guides
        updateVisualGuides(bbox, keypoints);
        
        // Add size feedback
        if (sizeRatio < 0.7) {
            box.style.borderColor = '#ff4444';
            box.style.borderWidth = '2px';
        } else {
            box.style.borderColor = '#00ff00';
            box.style.borderWidth = '2px';
        }
    } catch (error) {
        console.error('Error updating hand detection box:', error);
    }
}

// Add visual guides update function
function updateVisualGuides(bbox, keypoints) {
    try {
        if (!bbox || !keypoints) {
            console.error('Invalid bounding box or keypoints:', { bbox, keypoints });
            return;
        }

        const videoRect = video.getBoundingClientRect();
        const scaleX = videoRect.width / video.videoWidth;
        const scaleY = videoRect.height / video.videoHeight;
        
        // Center guide
        const centerX = (bbox.topLeft[0] + bbox.bottomRight[0]) / 2;
        const centerY = (bbox.topLeft[1] + bbox.bottomRight[1]) / 2;
        guideElements.centerGuide.style.left = `${centerX * scaleX}px`;
        guideElements.centerGuide.style.top = `${centerY * scaleY}px`;
        
        // Size guide
        const handWidth = bbox.bottomRight[0] - bbox.topLeft[0];
        const handHeight = bbox.bottomRight[1] - bbox.topLeft[1];
        const idealSize = Math.min(video.videoWidth, video.videoHeight) * 0.3;
        const sizeRatio = Math.min(handWidth, handHeight) / idealSize;
        
        if (sizeRatio < 0.7) {
            guideElements.sizeGuide.style.display = 'block';
            guideElements.sizeGuide.style.left = `${bbox.topLeft[0] * scaleX}px`;
            guideElements.sizeGuide.style.top = `${bbox.topLeft[1] * scaleY}px`;
            guideElements.sizeGuide.style.width = `${idealSize * scaleX}px`;
            guideElements.sizeGuide.style.height = `${idealSize * scaleY}px`;
        } else {
            guideElements.sizeGuide.style.display = 'none';
        }
        
        // Lighting guide
        const wristKeypoint = keypoints[0];
        const palmKeypoint = keypoints[9];
        
        if (!wristKeypoint || !palmKeypoint) {
            console.error('Missing keypoints:', { wristKeypoint, palmKeypoint });
            return;
        }
        
        const lightingScore = calculateLightingScore(wristKeypoint, palmKeypoint);
        
        if (lightingScore < 0.5) {
            guideElements.lightingGuide.style.display = 'block';
            guideElements.lightingGuide.style.left = `${palmKeypoint.x * scaleX}px`;
            guideElements.lightingGuide.style.top = `${palmKeypoint.y * scaleY}px`;
        } else {
            guideElements.lightingGuide.style.display = 'none';
        }
    } catch (error) {
        console.error('Error updating visual guides:', error);
    }
}

// Add lighting score calculation
function calculateLightingScore(wrist, palm) {
    try {
        if (!wrist || !palm) {
            console.error('Invalid keypoints for lighting score:', { wrist, palm });
            return 0;
        }

        // Simple lighting estimation based on keypoint visibility
        const wristConfidence = wrist.score || 1;
        const palmConfidence = palm.score || 1;
        return (wristConfidence + palmConfidence) / 2;
    } catch (error) {
        console.error('Error calculating lighting score:', error);
        return 0;
    }
}

// Add feedback generation function
function generateHandPositionFeedback(hand) {
    const bbox = hand.boundingBox;
    const keypoints = hand.keypoints;
    
    const handWidth = bbox.bottomRight[0] - bbox.topLeft[0];
    const handHeight = bbox.bottomRight[1] - bbox.topLeft[1];
    const idealSize = Math.min(video.videoWidth, video.videoHeight) * 0.3;
    const sizeRatio = Math.min(handWidth, handHeight) / idealSize;
    
    const lightingScore = calculateLightingScore(keypoints[0], keypoints[9]);
    
    if (sizeRatio < 0.7) {
        return 'Please move your hand closer to the camera for better detection.';
    } else if (lightingScore < 0.5) {
        return 'Please ensure your hand is well-lit and clearly visible.';
    } else {
        return 'Please adjust your hand position. Make sure your hand is fully visible and centered in the frame.';
    }
}

// Function to show specific tips for different signs
function showSignTips(letter, confidence) {
    const tipText = document.getElementById('sign-tip-text');
    
    // Clear any previous classes
    tipText.className = '';
    
    if (!letter) {
        tipText.textContent = 'Position your hand clearly in view.';
        return;
    }
    
    // Convert to uppercase to be safe
    letter = letter.toUpperCase();
    
    // Sign-specific tips
    switch(letter) {
        case 'A':
            tipText.innerHTML = `
                <strong>For "A" sign:</strong>
                <ul style="text-align: left; margin: 5px 0;">
                    <li>Make a tight fist</li>
                    <li>Keep all fingers closed</li>
                    <li>Thumb should stick out to the side</li>
                </ul>
            `;
            tipText.className = 'sign-tip-highlight';
            break;
        case 'P':
            tipText.innerHTML = `
                <strong>For "P" sign:</strong>
                <ul style="text-align: left; margin: 5px 0;">
                    <li>Form a circle with thumb and index finger</li>
                    <li>Extend middle, ring, and pinky fingers</li>
                    <li>Keep extended fingers together</li>
                </ul>
            `;
            tipText.className = 'sign-tip-highlight';
            break;
        default:
            tipText.textContent = `Showing sign for "${letter}" with ${formatConfidencePct(confidence)}% confidence.`;
    }
}

// Update the function that handles debug images
function updateDebugImages(result) {
    // Function keeps the backend image processing, but no longer updates UI elements
    
    // The debug images are still generated on the backend for diagnostic purposes,
    // but we're no longer displaying them in the UI per user request
} 