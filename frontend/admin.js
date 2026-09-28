const API_BASE = '/api';

const imageInputFiles = document.getElementById('imageInputFiles');
const imageInputFolder = document.getElementById('imageInputFolder');
const selectedInfo = document.getElementById('selectedInfo');
const fileCountText = document.getElementById('fileCount');
const uploadBtn = document.getElementById('uploadBtn');
const statusText = document.getElementById('statusText');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');
const liveActivity = document.getElementById('liveActivity');

const actReading = document.getElementById('act-reading');
const actExtracting = document.getElementById('act-extracting');
const actSimilarity = document.getElementById('act-similarity');
const actUploading = document.getElementById('act-uploading');
const actComplete = document.getElementById('act-complete');

const statImages = document.getElementById('stat-images');
const statEmbeddings = document.getElementById('stat-embeddings');
const statQdrant = document.getElementById('stat-qdrant');
const statTime = document.getElementById('stat-time');

let selectedFiles = [];

// Drag and drop support
const dropZone = document.getElementById('dropZoneFiles');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.background = 'rgba(255, 215, 0, 0.2)';
});
dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.background = 'rgba(255, 215, 0, 0.05)';
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.background = 'rgba(255, 215, 0, 0.05)';
    if(e.dataTransfer.files) handleFileSelection(e.dataTransfer.files);
});

function setStatus(message) {
    statusText.textContent = message;
}

function resetActivity() {
    [actReading, actExtracting, actSimilarity, actUploading, actComplete].forEach(el => el.classList.remove('done'));
}

function handleFileSelection(files) {
    selectedFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    if (selectedFiles.length > 0) {
        selectedInfo.style.display = 'block';
        fileCountText.textContent = selectedFiles.length;
        setStatus(`Ready to upload ${selectedFiles.length} images.`);
    } else {
        selectedInfo.style.display = 'none';
        setStatus('No images found in selection.');
    }
}

imageInputFiles.addEventListener('change', (e) => handleFileSelection(e.target.files));
imageInputFolder.addEventListener('change', (e) => handleFileSelection(e.target.files));

uploadBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        setStatus('Please select files or a folder first.');
        return;
    }

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('images', file);
    });

    uploadBtn.disabled = true;
    liveActivity.style.display = 'block';
    resetActivity();
    
    const startTime = Date.now();
    
    // Simulate initial progress
    progressBar.style.width = '10%';
    progressPercent.textContent = '10%';
    setStatus('Reading Images...');
    actReading.classList.add('done');

    try {
        // Fake progress for UX
        let simulatedProgress = 10;
        const progInt = setInterval(() => {
            if(simulatedProgress < 90) {
                simulatedProgress += Math.random() * 5;
                progressBar.style.width = `${Math.min(simulatedProgress, 90)}%`;
                progressPercent.textContent = `${Math.floor(Math.min(simulatedProgress, 90))}%`;
                
                if(simulatedProgress > 30 && !actExtracting.classList.contains('done')) {
                    actExtracting.classList.add('done');
                    setStatus('Extracting AI Features...');
                }
                if(simulatedProgress > 55 && !actSimilarity.classList.contains('done')) {
                    actSimilarity.classList.add('done');
                    setStatus('Checking Similarity Search...');
                }
                if(simulatedProgress > 75 && !actUploading.classList.contains('done')) {
                    actUploading.classList.add('done');
                    setStatus('Uploading to Qdrant...');
                }
            }
        }, 500);

        const response = await fetch(`${API_BASE}/upload_inventory`, {
            method: 'POST',
            body: formData,
        });
        
        clearInterval(progInt);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || error.details || 'Upload failed.');
        }

        const data = await response.json();
        const numUploaded = selectedFiles.length;
        
        progressBar.style.width = '100%';
        progressPercent.textContent = '100%';
        
        actExtracting.classList.add('done');
        actSimilarity.classList.add('done');
        actUploading.classList.add('done');
        actComplete.classList.add('done');
        setStatus(data.message || 'Completed Successfully!');
        
        // Update stats
        const timeTaken = Math.round((Date.now() - startTime) / 1000);
        statImages.textContent = numUploaded;
        statEmbeddings.textContent = numUploaded;
        statQdrant.textContent = numUploaded;
        statTime.textContent = `${timeTaken} sec`;

        // Reset selection
        selectedFiles = [];
        selectedInfo.style.display = 'none';
        imageInputFiles.value = '';
        imageInputFolder.value = '';

    } catch (error) {
        progressBar.style.width = '0%';
        progressPercent.textContent = '0%';
        progressBar.style.background = '#e74c3c';
        setStatus(`Error: ${error.message}`);
    } finally {
        uploadBtn.disabled = false;
    }
});
