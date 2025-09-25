// Fetal Hypoxia Detection System - Frontend JavaScript
// Handles file upload, API communication, and result display

class HypoxiaDetectionApp {
    constructor() {
        this.apiUrl = 'http://localhost:8000'; // Backend API URL
        this.heaFile = null;
        this.datFile = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkBackendStatus();
    }

    setupEventListeners() {
        // File upload zones
        this.setupFileUpload('hea', 'heaFile', 'heaUploadZone', 'heaFileName');
        this.setupFileUpload('dat', 'datFile', 'datUploadZone', 'datFileName');

        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeFiles();
        });
    }

    setupFileUpload(fileType, inputId, zoneId, fileNameId) {
        const input = document.getElementById(inputId);
        const zone = document.getElementById(zoneId);
        const fileName = document.getElementById(fileNameId);

        // Click to upload
        zone.addEventListener('click', () => input.click());

        // File selection
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this[fileType + 'File'] = file;
                fileName.textContent = file.name;
                fileName.classList.remove('hidden');
                zone.classList.add('border-green-300', 'bg-green-50');

                // Update upload zone icon
                const icon = zone.querySelector('i');
                icon.className = 'fas fa-check-circle text-4xl text-green-500 mb-4';

                this.checkUploadReady();
            }
        });

        // Drag and drop
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                const expectedExt = '.' + fileType;

                if (file.name.toLowerCase().endsWith(expectedExt)) {
                    input.files = files;
                    input.dispatchEvent(new Event('change'));
                } else {
                    this.showError(`Please upload a ${expectedExt.toUpperCase()} file`);
                }
            }
        });
    }

    checkUploadReady() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (this.heaFile && this.datFile) {
            // Check if filenames match (without extension)
            const heaName = this.heaFile.name.replace('.hea', '');
            const datName = this.datFile.name.replace('.dat', '');

            if (heaName === datName) {
                analyzeBtn.disabled = false;
                analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            } else {
                analyzeBtn.disabled = true;
                analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
                this.showError('HEA and DAT files must have matching names (e.g., 1001.hea and 1001.dat)');
            }
        } else {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
    }

    async checkBackendStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/`);
            const data = await response.json();

            if (data.status === 'healthy') {
                console.log('✅ Backend is healthy');
                console.log('📦 Available methods:', data.available_methods);
            }
        } catch (error) {
            console.warn('⚠️ Backend not accessible:', error.message);
            this.showError('Backend API is not accessible. Please make sure the server is running.');
        }
    }

    async analyzeFiles() {
        if (!this.heaFile || !this.datFile) {
            this.showError('Please upload both HEA and DAT files');
            return;
        }

        const method = document.querySelector('input[name="method"]:checked').value;

        try {
            this.showProgress();

            // Prepare form data
            const formData = new FormData();
            formData.append('hea_file', this.heaFile);
            formData.append('dat_file', this.datFile);
            formData.append('method', method);

            this.updateProgress(20, 'Uploading files...');

            // Send to backend
            const response = await fetch(`${this.apiUrl}/predict_complete`, {
                method: 'POST',
                body: formData
            });

            this.updateProgress(60, 'Processing files...');

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            this.updateProgress(80, 'Making prediction...');

            const result = await response.json();

            this.updateProgress(100, 'Analysis complete!');

            setTimeout(() => {
                this.displayResults(result);
            }, 500);

        } catch (error) {
            console.error('Analysis error:', error);
            this.hideProgress();
            this.showError(error.message);
        }
    }

    showProgress() {
        document.getElementById('progressSection').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');
        document.getElementById('errorSection').classList.add('hidden');

        // Scroll to progress section
        document.getElementById('progressSection').scrollIntoView({
            behavior: 'smooth'
        });
    }

    updateProgress(percentage, text) {
        document.getElementById('progressBar').style.width = percentage + '%';
        document.getElementById('progressText').textContent = text;
    }

    hideProgress() {
        document.getElementById('progressSection').classList.add('hidden');
    }

    displayResults(result) {
        this.hideProgress();

        // Show results section
        document.getElementById('resultsSection').classList.remove('hidden');
        document.getElementById('errorSection').classList.add('hidden');

        // Main prediction
        const predictionText = document.getElementById('predictionText');
        const predictionBadge = document.getElementById('predictionBadge');
        const confidenceText = document.getElementById('confidenceText');

        predictionText.textContent = result.prediction;
        confidenceText.textContent = (result.confidence * 100).toFixed(1) + '%';

        // Style prediction badge based on result
        predictionBadge.className = 'inline-flex items-center px-6 py-3 rounded-full text-lg font-semibold mb-4';

        if (result.prediction === 'Normal') {
            predictionBadge.classList.add('bg-green-100', 'text-green-800');
        } else if (result.prediction === 'Suspect') {
            predictionBadge.classList.add('bg-yellow-100', 'text-yellow-800');
        } else if (result.prediction === 'Hypoxia') {
            predictionBadge.classList.add('bg-red-100', 'text-red-800');
        }

        // Interpretation
        const interpretation = result.interpretation;
        document.getElementById('riskLevel').textContent = interpretation.risk_level;
        document.getElementById('recommendation').textContent = interpretation.recommendation;
        document.getElementById('methodUsed').textContent = interpretation.method_used;

        // Probability distribution
        const probabilities = result.probabilities;
        this.updateProbabilityBar('normal', probabilities.Normal || 0);
        this.updateProbabilityBar('suspect', probabilities.Suspect || 0);
        this.updateProbabilityBar('hypoxia', probabilities.Hypoxia || 0);

        // Clinical insights
        if (interpretation.insights && interpretation.insights.length > 0) {
            this.displayInsights(interpretation.insights);
        }

        // Processing info
        document.getElementById('recordId').textContent = result.record_id;
        document.getElementById('signalLength').textContent =
            result.processing_info.signal_length.toLocaleString() + ' samples';
        document.getElementById('duration').textContent =
            result.processing_info.duration_minutes.toFixed(1) + ' minutes';

        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({
            behavior: 'smooth'
        });
    }

    updateProbabilityBar(type, probability) {
        const percentage = (probability * 100);
        document.getElementById(type + 'Prob').style.width = percentage + '%';
        document.getElementById(type + 'Pct').textContent = percentage.toFixed(1) + '%';
    }

    displayInsights(insights) {
        const insightsSection = document.getElementById('insightsSection');
        const insightsList = document.getElementById('insightsList');

        insightsSection.classList.remove('hidden');
        insightsList.innerHTML = '';

        insights.forEach(insight => {
            const insightElement = document.createElement('div');
            insightElement.className = 'flex items-start mb-2 last:mb-0';
            insightElement.innerHTML = `
                <i class="fas fa-info-circle text-blue-500 mr-2 mt-0.5"></i>
                <span class="text-sm text-gray-700">${insight}</span>
            `;
            insightsList.appendChild(insightElement);
        });
    }

    showError(message) {
        document.getElementById('errorSection').classList.remove('hidden');
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('resultsSection').classList.add('hidden');

        // Scroll to error
        document.getElementById('errorSection').scrollIntoView({
            behavior: 'smooth'
        });

        // Auto-hide after 10 seconds
        setTimeout(() => {
            document.getElementById('errorSection').classList.add('hidden');
        }, 10000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new HypoxiaDetectionApp();
});