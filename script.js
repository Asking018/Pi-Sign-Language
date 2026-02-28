document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const loginModal = document.getElementById('login-modal');
    const loginForm = document.getElementById('login-form');
    const loginName = document.getElementById('login-name');
    const usernameDisplay = document.getElementById('username');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startCameraBtn = document.getElementById('start-camera');
    const switchCameraBtn = document.getElementById('switch-camera');
    const captureBtn = document.getElementById('capture');
    const translationOutput = document.getElementById('translation-output');
    const clearInputBtn = document.getElementById('clear-input');
    const addSpaceBtn = document.getElementById('add-space');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Variables
    let stream = null;
    let facingMode = 'user'; // Default to front camera
    let translatedText = '';
    let model = null;
    
    // Check if username is stored in localStorage
    const storedUsername = localStorage.getItem('signpi-username');
    if (!storedUsername) {
        // Show login modal if no username is stored
        loginModal.classList.add('active');
    } else {
        usernameDisplay.textContent = storedUsername;
    }
    
    // Handle login form submission
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const name = loginName.value.trim();
        if (name) {
            localStorage.setItem('signpi-username', name);
            usernameDisplay.textContent = name;
            loginModal.classList.remove('active');
        }
    });
    
    // Initialize the sign language translation model
    async function initModel() {
        try {
            loadingIndicator.classList.remove('hidden');
            
            // For a TensorFlow SavedModel format, use the model directory path
            // Assuming your model files are in a folder called 'sign_model'
            model = await tf.loadGraphModel('sign_model/model.json');
            
            // Or if you've converted to tfjs format specifically:
            // model = await tf.loadLayersModel('sign_model/model.json');
            
            console.log('Sign language model loaded successfully');
            loadingIndicator.classList.add('hidden');
            return true;
        } catch (error) {
            console.error('Failed to load model:', error);
            loadingIndicator.classList.add('hidden');
            alert('Could not load the sign language model. Please check console for details.');
            return false;
        }
    }
    
    // Start camera function
    async function startCamera() {
        try {
            if (stream) {
                stopCamera();
            }
            
            const constraints = {
                video: {
                    facingMode: facingMode
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            
            // Show switch camera button only if mobile device
            if (isMobileDevice()) {
                switchCameraBtn.classList.remove('hidden');
            }
            
            captureBtn.disabled = false;
            startCameraBtn.textContent = 'Restart Camera';
            
            return true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Could not access the camera. Please grant permission and try again.');
            return false;
        }
    }
    
    // Stop camera function
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            stream = null;
        }
    }
    
    // Switch camera function (front/back)
    function switchCamera() {
        facingMode = facingMode === 'user' ? 'environment' : 'user';
        startCamera();
    }
    
    // Capture frame from video for processing
    async function captureFrame() {
        if (!stream) {
            alert('Please start the camera first');
            return null;
        }
        
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw the current video frame to the canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Get image data for processing
        return canvas.toDataURL('image/jpeg');
    }
    
    // Process the captured frame with the model
    async function processFrame() {
        loadingIndicator.classList.remove('hidden');
        
        try {
            const imageData = await captureFrame();
            if (!imageData) {
                loadingIndicator.classList.add('hidden');
                return;
            }
            
            // Create an image element from the captured frame
            const img = new Image();
            img.src = imageData;
            
            img.onload = async function() {
                // Now process with the model
                if (model) {
                    const prediction = await model.predict(img);
                    translatedText += prediction;
                    translationOutput.textContent = translatedText;
                } else {
                    console.error('Model not loaded');
                    alert('Translation model is not ready. Please try again.');
                }
                loadingIndicator.classList.add('hidden');
            };
            
        } catch (error) {
            console.error('Error processing frame:', error);
            alert('Failed to process the sign. Please try again.');
            loadingIndicator.classList.add('hidden');
        }
    }
    
    // Clear input text
    function clearInput() {
        translatedText = '';
        translationOutput.textContent = '';
    }
    
    // Add space to input text
    function addSpace() {
        translatedText += ' ';
        translationOutput.textContent = translatedText;
    }
    
    // Check if the device is mobile using feature detection
    function isMobileDevice() {
        // Use matchMedia to check if the device has touch capabilities and a suitable viewport size
        return (('ontouchstart' in window) || (navigator.maxTouchPoints > 0)) && 
               window.matchMedia('(max-width: 768px)').matches;
    }
    
    // Event listeners
    startCameraBtn.addEventListener('click', startCamera);
    switchCameraBtn.addEventListener('click', switchCamera);
    captureBtn.addEventListener('click', processFrame);
    clearInputBtn.addEventListener('click', clearInput);
    addSpaceBtn.addEventListener('click', addSpace);
    
    // Initialize
    (async function initialize() {
        // Start loading the model
        const modelLoaded = await initModel();
        if (!modelLoaded) {
            alert('Failed to load sign language translation model. Some features may not work.');
        }
        
        // Disable capture button until camera starts
        captureBtn.disabled = true;
    })();
    
    // Cleanup
    window.addEventListener('beforeunload', function() {
        stopCamera();
    });
});