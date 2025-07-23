
// CV Optimizer Pro - Main JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let cvText = '';
    let isProcessing = false;
    let currentLanguage = 'pl';

    // Get DOM elements
    const cvUploadForm = document.getElementById('cv-upload-form');
    const cvFileInput = document.getElementById('cv-file');
    const cvTextInput = document.getElementById('cv-text');
    const jobTitleInput = document.getElementById('job-title');
    const jobDescriptionInput = document.getElementById('job-description');
    const jobUrlInput = document.getElementById('job-url');
    
    // Buttons
    const processButton = document.getElementById('process-button');
    const uploadBtn = document.getElementById('upload-btn');
    
    // Results and displays
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const processingIndicator = document.getElementById('processing-indicator');
    const uploadSuccessAlert = document.getElementById('upload-success');
    const uploadErrorAlert = document.getElementById('upload-error');
    const errorMessageSpan = document.getElementById('error-message');

    // CV Preview and Editor
    const cvPreview = document.getElementById('cv-preview');
    const cvEditor = document.getElementById('cv-editor');
    const cvTextDisplay = document.getElementById('cv-text-display');
    const cvTextEditor = document.getElementById('cv-text-editor');
    const editCvBtn = document.getElementById('edit-cv-btn');
    const saveCvBtn = document.getElementById('save-cv-btn');
    const cancelEditBtn = document.getElementById('cancel-edit-btn');

    // Language selector
    const languageSelector = document.getElementById('language-selector');
    if (languageSelector) {
        languageSelector.addEventListener('change', function() {
            currentLanguage = this.value;
        });
    }

    // Get analyze buttons if they exist
    const analyzeJobBtn = document.getElementById('analyze-job-btn');
    const analyzeUrlBtn = document.getElementById('analyze-url-btn');

    // Upload form handler
    if (cvUploadForm) {
        cvUploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            uploadCV();
        });
    }

    // Upload button handler
    if (uploadBtn) {
        uploadBtn.addEventListener('click', function(e) {
            e.preventDefault();
            uploadCV();
        });
    }

    // Process button handler
    if (processButton) {
        processButton.addEventListener('click', function(e) {
            e.preventDefault();
            processCV();
        });
    }

    // CV Editor handlers
    if (editCvBtn) {
        editCvBtn.addEventListener('click', function() {
            showCVEditor();
        });
    }

    if (saveCvBtn) {
        saveCvBtn.addEventListener('click', function() {
            saveCVChanges();
        });
    }

    if (cancelEditBtn) {
        cancelEditBtn.addEventListener('click', function() {
            cancelCVEdit();
        });
    }

    // Analyze job description button
    if (analyzeJobBtn) {
        analyzeJobBtn.addEventListener('click', function() {
            analyzeJobDescription();
        });
    }

    // Analyze URL button  
    if (analyzeUrlBtn) {
        analyzeUrlBtn.addEventListener('click', function() {
            extractFromJobUrl();
        });
    }

    // Upload CV function - optimized
    function uploadCV() {
        if (isProcessing) {
            showError('Operacja już w toku, proszę czekać...');
            return;
        }

        const formData = new FormData();
        let hasInput = false;
        
        // Check for file input
        if (cvFileInput?.files?.length > 0) {
            const file = cvFileInput.files[0];
            if (file.size > 16 * 1024 * 1024) { // 16MB limit check
                showError('Plik jest za duży. Maksymalny rozmiar to 16MB.');
                return;
            }
            formData.append('cv_file', file);
            hasInput = true;
        }
        
        // Check for text input
        if (cvTextInput?.value?.trim()) {
            formData.append('cv_text', cvTextInput.value.trim());
            hasInput = true;
        }
        
        if (!hasInput) {
            showError('Wybierz plik CV lub wklej tekst CV');
            return;
        }
        
        // Add optional fields
        if (jobTitleInput?.value) {
            formData.append('job_title', jobTitleInput.value);
        }
        
        if (jobDescriptionInput?.value) {
            formData.append('job_description', jobDescriptionInput.value);
        }

        isProcessing = true;
        showProcessing('Przesyłanie i przetwarzanie CV...');

        // Add timeout to prevent hanging requests
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

        fetch('/upload-cv', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                cvText = data.cv_text;
                showSuccess('✅ CV zostało pomyślnie przesłane!');
                displayCVPreview(cvText);
                enableProcessing();
            } else {
                showError(data.message || 'Błąd podczas przesyłania CV');
            }
        })
        .catch(error => {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                showError('Przesyłanie przekroczyło limit czasu. Spróbuj ponownie.');
            } else {
                console.error('Upload error:', error);
                showError(`Błąd: ${error.message}`);
            }
        })
        .finally(() => {
            isProcessing = false;
            hideProcessing();
        });
    }

    // Process CV function - optimized with timeout
    function processCV() {
        if (isProcessing) {
            showError('Analiza już w toku, proszę czekać...');
            return;
        }
        
        if (!cvText) {
            showError('Najpierw prześlij CV');
            return;
        }

        const selectedOption = document.querySelector('input[name="analysis_option"]:checked');
        if (!selectedOption) {
            showError('Wybierz rodzaj analizy');
            return;
        }

        isProcessing = true;
        const optionText = selectedOption.parentElement.textContent.trim();
        showProcessing(`🤖 Analizuję CV przez AI: ${optionText}...`);

        const requestData = {
            cv_text: cvText,
            selected_option: selectedOption.value,
            job_description: jobDescriptionInput?.value || '',
            job_title: jobTitleInput?.value || '',
            language: currentLanguage
        };

        // Add timeout for AI processing (2 minutes)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000);

        fetch('/process-cv', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData),
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                displayResult(data.result, selectedOption.value);
                showSuccess('✅ Analiza zakończona pomyślnie!');
            } else if (data.payment_required) {
                showPaymentRequired(data.message);
            } else if (data.premium_required) {
                showPremiumRequired(data.message);
            } else {
                showError(data.message || 'Błąd podczas przetwarzania CV');
            }
        })
        .catch(error => {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                showError('Analiza przekroczyła limit czasu. Spróbuj ponownie z krótszym CV.');
            } else {
                console.error('Processing error:', error);
                showError(`Błąd przetwarzania: ${error.message}`);
            }
        })
        .finally(() => {
            isProcessing = false;
            hideProcessing();
        });
    }

    // Analyze job description - optimized
    function analyzeJobDescription() {
        const jobDescription = jobDescriptionInput?.value?.trim();
        
        if (!jobDescription) {
            showError('Wprowadź opis stanowiska do analizy');
            return;
        }

        if (jobDescription.length < 50) {
            showError('Opis stanowiska jest za krótki (minimum 50 znaków)');
            return;
        }

        showProcessing('🔍 Analizuję opis stanowiska...');

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 1 minute timeout

        fetch('/analyze-job-posting', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                job_description: jobDescription,
                language: currentLanguage
            }),
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeoutId);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.success) {
                displayJobAnalysis(data.analysis);
                showSuccess('✅ Analiza stanowiska zakończona!');
            } else {
                showError(data.message || 'Błąd podczas analizy stanowiska');
            }
        })
        .catch(error => {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                showError('Analiza przekroczyła limit czasu');
            } else {
                console.error('Job analysis error:', error);
                showError(`Błąd analizy: ${error.message}`);
            }
        })
        .finally(() => {
            hideProcessing();
        });
    }

    // Extract from job URL
    function extractFromJobUrl() {
        const jobUrl = jobUrlInput ? jobUrlInput.value.trim() : '';
        
        if (!jobUrl) {
            showError('Wprowadź URL oferty pracy');
            return;
        }

        showProcessing('Pobieranie opisu z URL...');

        fetch('/analyze-job-posting', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                job_url: jobUrl,
                language: currentLanguage
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (jobDescriptionInput) {
                    jobDescriptionInput.value = data.raw_description || '';
                }
                showSuccess('Opis został pobrany z URL!');
            } else {
                showError(data.message || 'Błąd podczas pobierania z URL');
            }
        })
        .catch(error => {
            console.error('URL extraction error:', error);
            showError('Wystąpił błąd podczas pobierania z URL');
        })
        .finally(() => {
            hideProcessing();
        });
    }

    // Display functions
    function displayCVPreview(text) {
        if (cvPreview) {
            cvPreview.style.display = 'block';
        }
        if (cvTextDisplay) {
            cvTextDisplay.textContent = text;
        }
    }

    function displayResult(result, optionType) {
        if (!resultContainer || !resultText) return;
        
        resultContainer.style.display = 'block';
        
        // Clear previous result first
        resultText.innerHTML = '<div class="spinner-border spinner-border-sm me-2" role="status"></div>Formatowanie wyniku...';
        
        // Use setTimeout to prevent UI blocking
        setTimeout(() => {
            try {
                if (typeof result === 'object') {
                    resultText.innerHTML = formatObjectResult(result, optionType);
                } else {
                    resultText.innerHTML = formatTextResult(result);
                }
                
                // Scroll to result
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } catch (error) {
                resultText.innerHTML = `<div class="alert alert-warning">Błąd formatowania: ${error.message}</div>`;
            }
        }, 100);
    }

    function displayJobAnalysis(analysis) {
        // Create or update job analysis display
        let analysisContainer = document.getElementById('job-analysis-container');
        if (!analysisContainer) {
            analysisContainer = document.createElement('div');
            analysisContainer.id = 'job-analysis-container';
            analysisContainer.className = 'card mt-3';
            if (jobDescriptionInput && jobDescriptionInput.parentNode) {
                jobDescriptionInput.parentNode.appendChild(analysisContainer);
            }
        }

        analysisContainer.innerHTML = `
            <div class="card-header">
                <h5 class="mb-0">Analiza stanowiska</h5>
            </div>
            <div class="card-body">
                <pre style="white-space: pre-wrap; font-family: inherit;">${JSON.stringify(analysis, null, 2)}</pre>
            </div>
        `;
    }

    function formatObjectResult(result, optionType) {
        let html = '<div class="result-content">';
        
        if (optionType === 'cv_score' && result.score) {
            html += `<div class="score-display">
                <h4>Ocena CV: ${result.score}/100</h4>
                <div class="progress mb-3">
                    <div class="progress-bar" style="width: ${result.score}%"></div>
                </div>
            </div>`;
        }

        if (result.analysis) {
            html += `<div class="analysis-section">
                <h5>Analiza:</h5>
                <p>${result.analysis}</p>
            </div>`;
        }

        if (result.recommendations) {
            html += `<div class="recommendations-section">
                <h5>Rekomendacje:</h5>
                <ul>`;
            result.recommendations.forEach(rec => {
                html += `<li>${rec}</li>`;
            });
            html += `</ul></div>`;
        }

        if (result.optimized_cv || result.improved_cv) {
            html += `<div class="optimized-cv-section">
                <h5>Zoptymalizowane CV:</h5>
                <pre style="white-space: pre-wrap; font-family: inherit;">${result.optimized_cv || result.improved_cv}</pre>
            </div>`;
        }

        html += '</div>';
        return html;
    }

    function formatTextResult(text) {
        return `<pre style="white-space: pre-wrap; font-family: inherit;">${text}</pre>`;
    }

    // CV Editor functions
    function showCVEditor() {
        if (cvPreview) cvPreview.style.display = 'none';
        if (cvEditor) cvEditor.style.display = 'block';
        if (cvTextEditor) cvTextEditor.value = cvText;
    }

    function saveCVChanges() {
        if (cvTextEditor) {
            cvText = cvTextEditor.value;
            displayCVPreview(cvText);
        }
        if (cvEditor) cvEditor.style.display = 'none';
        if (cvPreview) cvPreview.style.display = 'block';
        showSuccess('Zmiany zostały zapisane');
    }

    function cancelCVEdit() {
        if (cvEditor) cvEditor.style.display = 'none';
        if (cvPreview) cvPreview.style.display = 'block';
    }

    // Utility functions
    function showSuccess(message) {
        if (uploadSuccessAlert) {
            uploadSuccessAlert.textContent = message;
            uploadSuccessAlert.style.display = 'block';
            setTimeout(() => {
                uploadSuccessAlert.style.display = 'none';
            }, 5000);
        }
    }

    function showError(message) {
        if (uploadErrorAlert) {
            uploadErrorAlert.style.display = 'block';
        }
        if (errorMessageSpan) {
            errorMessageSpan.textContent = message;
        }
        setTimeout(() => {
            if (uploadErrorAlert) {
                uploadErrorAlert.style.display = 'none';
            }
        }, 5000);
    }

    function showProcessing(message) {
        if (processingIndicator) {
            processingIndicator.style.display = 'block';
            const messageEl = processingIndicator.querySelector('.processing-message');
            if (messageEl) {
                messageEl.textContent = message;
            }
        }
    }

    function hideProcessing() {
        if (processingIndicator) {
            processingIndicator.style.display = 'none';
        }
    }

    function enableProcessing() {
        if (processButton) {
            processButton.disabled = false;
        }
        // Show analysis options
        const analysisOptions = document.getElementById('analysis-options');
        if (analysisOptions) {
            analysisOptions.style.display = 'block';
        }
    }

    function showPaymentRequired(message) {
        // Redirect to payment page or show payment modal
        alert(message + '\n\nZostaniesz przekierowany do strony płatności.');
        window.location.href = '/payment-options';
    }

    function showPremiumRequired(message) {
        // Redirect to premium subscription page
        alert(message + '\n\nZostaniesz przekierowany do subskrypcji Premium.');
        window.location.href = '/premium-subscription';
    }

    // Initialize payment processing if on checkout page
    if (window.location.pathname.includes('/checkout')) {
        initializeStripeCheckout();
    }

    // Copy result to clipboard
    document.addEventListener('click', function(e) {
        if (e.target.id === 'copy-result-btn' || e.target.closest('#copy-result-btn')) {
            const resultText = document.getElementById('result-text');
            if (resultText) {
                navigator.clipboard.writeText(resultText.textContent).then(() => {
                    showSuccess('Wynik skopiowany do schowka!');
                }).catch(() => {
                    showError('Nie udało się skopiować wyniku');
                });
            }
        }
    });
});

// Stripe checkout initialization
function initializeStripeCheckout() {
    const stripePublicKey = document.querySelector('meta[name="stripe-public-key"]');
    if (!stripePublicKey) return;

    const stripe = Stripe(stripePublicKey.content);
    const elements = stripe.elements();

    // Create card element
    const cardElement = elements.create('card', {
        style: {
            base: {
                fontSize: '16px',
                color: '#424770',
                '::placeholder': {
                    color: '#aab7c4',
                },
            },
        },
    });

    const cardContainer = document.getElementById('card-element');
    if (cardContainer) {
        cardElement.mount('#card-element');
    }

    // Handle form submission
    const paymentForm = document.getElementById('payment-form');
    if (paymentForm) {
        paymentForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            const {error, paymentIntent} = await stripe.confirmCardPayment(
                paymentForm.dataset.clientSecret,
                {
                    payment_method: {
                        card: cardElement,
                    }
                }
            );

            if (error) {
                console.error('Payment failed:', error);
                showError('Płatność nie powiodła się: ' + error.message);
            } else {
                console.log('Payment succeeded:', paymentIntent);
                // Verify payment on server
                verifyPayment(paymentIntent.id);
            }
        });
    }
}

// Verify payment with server
function verifyPayment(paymentIntentId) {
    fetch('/verify-payment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            payment_intent_id: paymentIntentId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = '/payment-success';
        } else {
            showError(data.message || 'Błąd weryfikacji płatności');
        }
    })
    .catch(error => {
        console.error('Payment verification error:', error);
        showError('Błąd podczas weryfikacji płatności');
    });
}
