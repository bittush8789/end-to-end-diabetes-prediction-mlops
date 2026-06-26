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
            
            resultCard.style.display = 'none';
            document.querySelectorAll('.error-message').forEach(el => el.remove());
            
            const formData = new FormData(form);
            const data = {};
            let hasErrors = false;

            for (const [key, value] of formData.entries()) {
                const numVal = parseFloat(value);
                const limit = limits[key];
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
                const firstError = document.querySelector('.error-message');
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                return;
            }

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
                    throw new Error(result.detail || 'Inference failed');
                }

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

        statusEl.textContent = res.prediction;
        if (res.prediction === 'Diabetic') {
            statusEl.className = 'result-status diabetic';
        } else {
            statusEl.className = 'result-status non-diabetic';
        }

        const percentage = (res.probability * 100).toFixed(1);
        probEl.textContent = `${percentage}%`;

        // Classification risk level
        let riskLevel = 'Low Risk';
        if (res.probability >= 0.40 && res.probability <= 0.70) {
            riskLevel = 'Medium Risk';
        } else if (res.probability > 0.70) {
            riskLevel = 'High Risk';
        }
        
        riskEl.textContent = riskLevel;

        progressBar.style.width = `${percentage}%`;
        progressBar.className = 'progress-bar';
        if (riskLevel === 'Low Risk') {
            progressBar.classList.add('low-risk');
            riskEl.style.color = 'var(--success)';
        } else if (riskLevel === 'Medium Risk') {
            progressBar.classList.add('medium-risk');
            riskEl.style.color = 'var(--warning)';
        } else {
            progressBar.classList.add('high-risk');
            riskEl.style.color = 'var(--danger)';
        }

        let tipsHtml = '';
        if (res.prediction === 'Diabetic') {
            tipsHtml += '<li><strong>Physician Consult Required:</strong> Make an appointment with a doctor for checkups.</li>';
        } else {
            tipsHtml += '<li><strong>Healthy Diet:</strong> Maintain intake of dietary fiber and slow-release carbohydrates.</li>';
        }

        if (inputData.BMI > 25) {
            tipsHtml += `<li><strong>Active Lifestyle (BMI: ${inputData.BMI}):</strong> Work towards reducing weight through active workouts.</li>`;
        }
        if (inputData.Glucose > 140) {
            tipsHtml += `<li><strong>Dietary Control (Glucose: ${inputData.Glucose} mg/dL):</strong> Avoid sugars and check glucose levels regularly.</li>`;
        }
        if (inputData.BloodPressure > 80) {
            tipsHtml += `<li><strong>Hypertension Warning (BP: ${inputData.BloodPressure} mmHg):</strong> Reduce sodium and track diastolic pressures.</li>`;
        }
        
        tipsHtml += '<li><strong>Routine Diagnostics:</strong> Test blood glucose levels annually to track any progression.</li>';
        tipsEl.innerHTML = tipsHtml;

        resultCard.style.display = 'block';
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
