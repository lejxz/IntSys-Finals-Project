const scanButton = document.getElementById('scanBtn');
const emailText = document.getElementById('emailText');
const results = document.getElementById('results');
const errorText = document.getElementById('errorText');

const safeValue = document.getElementById('safeValue');
const fraudValue = document.getElementById('fraudValue');
const injectionValue = document.getElementById('injectionValue');
const safeBar = document.getElementById('safeBar');
const fraudBar = document.getElementById('fraudBar');
const injectionBar = document.getElementById('injectionBar');
const statusBadge = document.getElementById('statusBadge');
const fraudType = document.getElementById('fraudType');
const injectionType = document.getElementById('injectionType');
const predictedLabel = document.getElementById('predictedLabel');

function toPercent(value) {
  return Math.round((Number(value) || 0) * 100);
}

function updateResultUI(payload) {
  const safePercent = toPercent(payload.is_safe);
  const fraudPercent = toPercent(payload.is_fraud);
  const injectionPercent = toPercent(payload.is_injection);

  safeValue.textContent = `${safePercent}%`;
  fraudValue.textContent = `${fraudPercent}%`;
  injectionValue.textContent = `${injectionPercent}%`;

  safeBar.style.width = `${safePercent}%`;
  fraudBar.style.width = `${fraudPercent}%`;
  injectionBar.style.width = `${injectionPercent}%`;

  statusBadge.textContent = payload.status || '-';
  statusBadge.classList.remove('safe', 'malicious');
  statusBadge.classList.add((payload.status || '').toLowerCase());

  fraudType.textContent = payload.predicted_label === 'fraud' ? 'fraud' : 'None detected';
  injectionType.textContent = payload.predicted_label === 'injection' ? 'injection' : 'None detected';
  predictedLabel.textContent = `Predicted class: ${payload.predicted_label || 'unknown'}`;
}

async function scanEmail() {
  errorText.classList.add('hidden');
  const text = emailText.value.trim();

  if (!text) {
    errorText.textContent = 'Please enter email content before scanning.';
    errorText.classList.remove('hidden');
    return;
  }

  scanButton.disabled = true;
  scanButton.textContent = 'Scanning...';

  try {
    const response = await fetch('/scan', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Scan failed.');
    }

    updateResultUI(data);
    results.classList.remove('hidden');
  } catch (error) {
    errorText.textContent = error.message;
    errorText.classList.remove('hidden');
    results.classList.add('hidden');
  } finally {
    scanButton.disabled = false;
    scanButton.textContent = 'Scan Email';
  }
}

scanButton.addEventListener('click', scanEmail);