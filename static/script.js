document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const form = document.getElementById('agronomyForm');
  const submitBtn = document.getElementById('submitBtn');
  const resetBtn = document.getElementById('resetBtn');
  const presetContainer = document.getElementById('presetButtons');
  const emptyPrompt = document.getElementById('emptyPrompt');
  const resultsWrapper = document.getElementById('resultsWrapper');
  const printReportBtn = document.getElementById('printReportBtn');

  // Input pairs
  const inputs = [
    { num: 'inputN', range: 'rangeN' },
    { num: 'inputP', range: 'rangeP' },
    { num: 'inputK', range: 'rangeK' },
    { num: 'inputPH', range: 'rangePH', isPH: true },
    { num: 'inputTemp', range: 'rangeTemp' },
    { num: 'inputHumidity', range: 'rangeHumidity' },
    { num: 'inputRain', range: 'rangeRain' }
  ];

  // Two-way synchronization between number inputs and sliders
  inputs.forEach(pair => {
    const numEl = document.getElementById(pair.num);
    const rangeEl = document.getElementById(pair.range);

    if (numEl && rangeEl) {
      numEl.addEventListener('input', () => {
        rangeEl.value = numEl.value;
        if (pair.isPH) updatePHLabel(parseFloat(numEl.value));
      });

      rangeEl.addEventListener('input', () => {
        numEl.value = rangeEl.value;
        if (pair.isPH) updatePHLabel(parseFloat(rangeEl.value));
      });
    }
  });

  function updatePHLabel(val) {
    const phTag = document.getElementById('phTag');
    if (!phTag || isNaN(val)) return;

    let desc = "Neutral";
    if (val < 5.5) desc = "Strongly Acidic";
    else if (val < 6.5) desc = "Slightly Acidic";
    else if (val <= 7.5) desc = "Neutral (Optimal)";
    else if (val <= 8.5) desc = "Slightly Alkaline";
    else desc = "Strongly Alkaline";

    phTag.textContent = `${val.toFixed(1)} (${desc})`;
  }

  // Fetch and display regional presets
  async function loadPresets() {
    try {
      const res = await fetch('/api/presets');
      const data = await res.json();

      if (data.status === 'success' && data.presets) {
        presetContainer.innerHTML = '';
        data.presets.forEach((preset, index) => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = `preset-btn ${index === 0 ? 'active' : ''}`;
          btn.innerHTML = `<strong>${preset.title.split('(')[0].trim()}</strong> <small style="opacity:0.75">(${preset.region.split('(')[0].trim()})</small>`;
          btn.title = preset.description;

          btn.addEventListener('click', () => {
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            applyPresetValues(preset.values);
            // Auto submit to evaluate immediately
            evaluateSoil();
          });

          presetContainer.appendChild(btn);
        });
      }
    } catch (err) {
      console.warn("Could not load presets dynamically:", err);
    }
  }

  function applyPresetValues(vals) {
    for (const [key, val] of Object.entries(vals)) {
      const inputEl = document.querySelector(`input[name="${key}"]`);
      if (inputEl) {
        inputEl.value = val;
        // Trigger sync to range slider
        const rangeEl = document.getElementById('range' + key.toUpperCase());
        if (rangeEl) rangeEl.value = val;
        if (key.toLowerCase() === 'ph') updatePHLabel(parseFloat(val));
      }
    }
  }

  // Handle Form Submission
  async function evaluateSoil() {
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>⏳ Evaluating Soil Matrix...</span>`;

    const payload = {
      N: parseFloat(document.getElementById('inputN').value),
      P: parseFloat(document.getElementById('inputP').value),
      K: parseFloat(document.getElementById('inputK').value),
      ph: parseFloat(document.getElementById('inputPH').value),
      temperature: parseFloat(document.getElementById('inputTemp').value),
      humidity: parseFloat(document.getElementById('inputHumidity').value),
      rainfall: parseFloat(document.getElementById('inputRain').value)
    };

    try {
      const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (result.status === 'success') {
        renderResults(result);
      } else {
        alert("Evaluation failed: " + (result.message || "Unknown error"));
      }
    } catch (err) {
      console.error(err);
      alert("Error contacting the agronomy server. Ensure app.py is running.");
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span>🌾 Run Soil Compatibility Evaluation</span>`;
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    evaluateSoil();
  });

  resetBtn.addEventListener('click', () => {
    applyPresetValues({
      N: 90, P: 42, K: 43, ph: 6.5,
      temperature: 25.0, humidity: 82.0, rainfall: 240.0
    });
    emptyPrompt.style.display = 'block';
    resultsWrapper.style.display = 'none';
    if (printReportBtn) printReportBtn.style.display = 'none';
  });

  // Render Result Payload to UI
  function renderResults(data) {
    emptyPrompt.style.display = 'none';
    resultsWrapper.style.display = 'block';
    if (printReportBtn) printReportBtn.style.display = 'inline-block';

    const p = data.prediction;
    const diag = data.soil_health_assessment;
    const bench = data.crop_benchmark || {};
    const input = data.input_received || {};

    // Hero fields
    document.getElementById('resCategory').textContent = p.category;
    document.getElementById('resCropName').textContent = p.display_name;
    document.getElementById('resConfidence').textContent = `${p.confidence}%`;
    document.getElementById('resSeason').textContent = p.season;
    document.getElementById('resDuration').textContent = p.growth_duration;
    document.getElementById('resWater').textContent = p.water_requirement;
    document.getElementById('resSoil').textContent = p.soil_suitability;
    document.getElementById('resAdvisory').textContent = p.fertilizer_guidance;

    // Diagnostic observations
    const diagList = document.getElementById('diagnosticList');
    diagList.innerHTML = '';
    
    // Add pH observation
    const phLi = document.createElement('li');
    phLi.className = 'diagnostic-item';
    phLi.innerHTML = `<span>📍</span> <span><strong>Soil Reaction:</strong> pH ${diag.ph_value.toFixed(1)} is classified as ${diag.ph_classification}.</span>`;
    diagList.appendChild(phLi);

    if (diag.diagnostics && diag.diagnostics.length > 0) {
      diag.diagnostics.forEach(note => {
        const li = document.createElement('li');
        li.className = 'diagnostic-item';
        li.innerHTML = `<span>🔹</span> <span>${note}</span>`;
        diagList.appendChild(li);
      });
    }

    // Benchmark comparison rows
    const tbody = document.getElementById('benchmarkRows');
    tbody.innerHTML = '';

    const params = [
      { name: 'Nitrogen (N)', key: 'N', benchKey: 'avg_N', unit: 'kg/ha' },
      { name: 'Phosphorus (P)', key: 'P', benchKey: 'avg_P', unit: 'kg/ha' },
      { name: 'Potassium (K)', key: 'K', benchKey: 'avg_K', unit: 'kg/ha' },
      { name: 'Soil pH', key: 'ph', benchKey: 'avg_ph', unit: '' },
      { name: 'Temperature', key: 'temperature', benchKey: 'avg_temp', unit: '°C' },
      { name: 'Humidity', key: 'humidity', benchKey: 'avg_humidity', unit: '%' },
      { name: 'Rainfall', key: 'rainfall', benchKey: 'avg_rainfall', unit: 'mm' }
    ];

    params.forEach(item => {
      const fieldVal = input[item.key] !== undefined ? input[item.key] : '-';
      const idealVal = bench[item.benchKey] !== undefined ? bench[item.benchKey] : '-';
      
      let statusTag = `<span class="status-tag optimal">Optimal</span>`;
      if (typeof fieldVal === 'number' && typeof idealVal === 'number') {
        const diff = fieldVal - idealVal;
        if (Math.abs(diff) > 20 && item.key !== 'ph') {
          if (diff > 0) {
            statusTag = `<span class="status-tag high">High (+${Math.round(diff)})</span>`;
          } else {
            statusTag = `<span class="status-tag deficient">Deficient (${Math.round(diff)})</span>`;
          }
        }
      }

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${item.name}</strong></td>
        <td>${fieldVal} ${item.unit}</td>
        <td>${idealVal} ${item.unit}</td>
        <td>${statusTag}</td>
      `;
      tbody.appendChild(tr);
    });

    // Secondary Alternatives
    const altContainer = document.getElementById('alternativesList');
    altContainer.innerHTML = '';
    if (data.alternatives && data.alternatives.length > 0) {
      data.alternatives.forEach(alt => {
        const div = document.createElement('div');
        div.className = 'alt-row';
        div.innerHTML = `
          <div>
            <div class="alt-name">${alt.display_name}</div>
            <div style="font-size:0.75rem; color:#64748b;">${alt.season}</div>
          </div>
          <div style="display:flex; align-items:center; gap:0.6rem;">
            <div class="alt-prob-bar-container">
              <div class="alt-prob-bar" style="width: ${Math.min(100, alt.probability * 1.5)}%;"></div>
            </div>
            <div class="alt-prob-text">${alt.probability}%</div>
          </div>
        `;
        altContainer.appendChild(div);
      });
    } else {
      altContainer.innerHTML = `<div style="font-size:0.8rem; color:#94a3b8;">No secondary candidates within viable threshold. Primary recommendation is dominant.</div>`;
    }
  }

  // Initial load
  loadPresets();
  // Run initial evaluation for default preset
  evaluateSoil();
});
