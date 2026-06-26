document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultCard = document.getElementById('result-card');
    const spinnerContainer = document.getElementById('spinner-container');
    const resetBtn = document.getElementById('reset-btn');

    // Validation thresholds
    const limits = {
        Pregnancies: { min: 0, max: 20 },
        Glucose: { min: 0, max: 300 },
        BloodPressure: { min: 0, max: 200 },
        SkinThickness: { min: 0, max: 100 },
        Insulin: { min: 0, max: 1000 },
        BMI: { min: 0, max: 80 },
        DiabetesPedigreeFunction: { min: 0, max: 3 },
        Age: { min: 0, max: 120 }
    };

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Clear previous results and error messages
            resultCard.style.display = 'none';
            document.querySelectorAll('.error-message').forEach(el => el.remove());
            
            // Get form data
            const formData = new FormData(form);
            const data = {};
            let hasErrors = false;

            for (const [key, value] of formData.entries()) {
                const numVal = parseFloat(value);
                const limit = limits[key];
                
                // Client-side validations
                const inputElement = document.getElementById(key);
                
                if (value.trim() === '') {
                    showError(inputElement, 'This field is required');
                    hasErrors = true;
                } else if (isNaN(numVal)) {
                    showError(inputElement, 'Must be a valid number');
                    hasErrors = true;
                } else if (numVal < 0) {
                    showError(inputElement, 'Value cannot be negative');
                    hasErrors = true;
                } else if (limit && (numVal < limit.min || numVal > limit.max)) {
                    showError(inputElement, `Value must be between ${limit.min} and ${limit.max}`);
                    hasErrors = true;
                } else {
                    data[key] = numVal;
                }
            }

            if (hasErrors) {
                // Scroll to first error
                const firstError = document.querySelector('.error-message');
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                return;
            }

            // Show loading spinner
            spinnerContainer.style.display = 'flex';
            spinnerContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                spinnerContainer.style.display = 'none';

                if (!response.ok) {
                    throw new Error(result.error || 'Server error occurred');
                }

                // Render result card details
                displayResults(result, data);

            } catch (err) {
                spinnerContainer.style.display = 'none';
                alert(`Error: ${err.message}`);
            }
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            form.reset();
            resultCard.style.display = 'none';
            document.querySelectorAll('.error-message').forEach(el => el.remove());
        });
    }

    function showError(element, message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.color = '#ef4444';
        errorDiv.style.fontSize = '0.8rem';
        errorDiv.style.marginTop = '0.25rem';
        errorDiv.textContent = message;
        element.parentElement.appendChild(errorDiv);
    }

    function displayResults(res, inputData) {
        const statusEl = document.getElementById('res-status');
        const probEl = document.getElementById('res-probability');
        const riskEl = document.getElementById('res-risk');
        const progressBar = document.getElementById('res-progress-bar');
        const tipsEl = document.getElementById('res-tips');

        // Set prediction text and styling
        statusEl.textContent = res.prediction;
        if (res.prediction === 'Diabetic') {
            statusEl.className = 'result-status diabetic';
        } else {
            statusEl.className = 'result-status non-diabetic';
        }

        // Set probability and risk level
        const percentage = (res.probability * 100).toFixed(1);
        probEl.textContent = `${percentage}%`;
        riskEl.textContent = res.risk_level;

        // Set progress bar
        progressBar.style.width = `${percentage}%`;
        progressBar.className = 'progress-bar'; // reset classes
        if (res.risk_level === 'Low Risk') {
            progressBar.classList.add('low-risk');
        } else if (res.risk_level === 'Medium Risk') {
            progressBar.classList.add('medium-risk');
        } else {
            progressBar.classList.add('high-risk');
        }

        // Generate customized suggestions
        let tipsHtml = '';
        if (res.prediction === 'Diabetic') {
            tipsHtml += '<li><strong>Consult a Physician:</strong> Schedule an appointment with an endocrinologist or primary care physician for a formal diagnostic evaluation.</li>';
        } else {
            tipsHtml += '<li><strong>Maintain a Balanced Diet:</strong> Keep eating nutrient-dense foods, rich in fiber and low in refined sugars.</li>';
        }

        if (inputData.BMI > 25) {
            tipsHtml += `<li><strong>Manage Weight (BMI: ${inputData.BMI}):</strong> Incorporate physical activities to help reduce BMI toward the normal range (18.5 - 24.9).</li>`;
        }
        if (inputData.Glucose > 140) {
            tipsHtml += `<li><strong>Monitor Glucose (Glucose: ${inputData.Glucose} mg/dL):</strong> High plasma glucose is a strong indicator. Try reducing carbohydrate intake and re-testing regularly.</li>`;
        }
        if (inputData.BloodPressure > 80) {
            tipsHtml += `<li><strong>Cardiovascular Health (BP: ${inputData.BloodPressure} mmHg):</strong> Track your blood pressure regularly. Reduce sodium intake and engage in aerobic exercises.</li>`;
        }
        
        tipsHtml += '<li><strong>Regular Screening:</strong> Maintain annual physicals and routine blood checks, particularly if you have family history.</li>';
        
        tipsEl.innerHTML = tipsHtml;

        // Display the card and scroll to it
        resultCard.style.display = 'block';
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
