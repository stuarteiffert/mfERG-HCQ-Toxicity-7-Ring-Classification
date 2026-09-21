// Must match the onnxruntime-web version of the <script> tag in index.html.
const ORT_VERSION = '1.18.0';

// Binary decision threshold applied to P(toxic), taken from external validation
// of this model.
//
// NOTE: this deliberately differs from the model's own `label` output, which is
// a fixed 0.5 argmax (scikit-learn's untuned default) baked into the ONNX
// TreeEnsembleClassifier and not changeable. The two disagree whenever
// P(toxic) falls between this threshold and 0.5, so the interface derives its
// verdict from the probability and ignores `label`.
const TOXIC_THRESHOLD = 0.4;

// Results lying within this fractional distance of the threshold are reported
// as indeterminate rather than forced into a binary verdict. It also fixes the
// confidence value at which a result becomes "Uncertain":
// 0.25 -> band 0.30-0.55, i.e. confidence <= 62.5%.
const UNCERTAIN_FRACTION = 0.25;

// Rounded to whole percent: the forest has 100 trees, so P(toxic) only ever
// takes values in 0.01 steps, and unrounded bounds (0.30000000000000004) would
// otherwise exclude a probability sitting exactly on the edge of the band.
const UNCERTAIN_LOWER = Math.round(TOXIC_THRESHOLD * (1 - UNCERTAIN_FRACTION) * 100) / 100;
const UNCERTAIN_UPPER = Math.round((TOXIC_THRESHOLD + (1 - TOXIC_THRESHOLD) * UNCERTAIN_FRACTION) * 100) / 100;

let session;
let scalerParams;

const form = document.getElementById('prediction-form');
const predictBtn = document.getElementById('predict-btn');
const resultSection = document.getElementById('result-section');
const predictionLabel = document.getElementById('prediction-label');
const resultIndicator = document.getElementById('result-indicator');
const resultDetails = document.getElementById('result-details');
const confidenceFill = document.getElementById('confidence-fill');
const statusMessage = document.getElementById('status-message');

/**
 * Show a human readable failure reason instead of a generic error, so that
 * users (and bug reports) carry the actual cause.
 */
function showError(title, advice, detail) {
    statusMessage.classList.remove('hidden');
    statusMessage.innerHTML = '';

    const heading = document.createElement('strong');
    heading.textContent = title;
    statusMessage.appendChild(heading);

    const text = document.createElement('span');
    text.textContent = advice;
    statusMessage.appendChild(text);

    if (detail) {
        const code = document.createElement('code');
        code.textContent = detail;
        statusMessage.appendChild(code);
    }
}

/**
 * Feature detection run before ONNX Runtime is asked to do anything.
 * Returns null when the browser is usable, otherwise a {title, advice} object.
 *
 * The SIMD probe is the same module that ONNX Runtime itself validates.
 * ONNX Runtime >= 1.19 is SIMD-only, and Safari only gained WebAssembly SIMD
 * in 16.4, which is why this app is pinned to 1.18.0 (the last release that
 * still ships a non-SIMD build).
 */
function checkBrowserSupport() {
    if (typeof WebAssembly !== 'object' || typeof WebAssembly.validate !== 'function') {
        return {
            title: 'WebAssembly is unavailable in this browser.',
            advice: 'This calculator runs the model locally using WebAssembly. ' +
                    'If you are using Safari, check whether Lockdown Mode is enabled ' +
                    '(Settings > Privacy & Security > Lockdown Mode), as it disables ' +
                    'WebAssembly. Otherwise please try Chrome, Edge or Firefox.'
        };
    }

    const simdSupported = WebAssembly.validate(new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 30, 1, 28, 0,
        65, 0, 253, 15, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        253, 186, 1, 26, 11
    ]));

    if (!simdSupported) {
        console.warn('WebAssembly SIMD is not available; falling back to the non-SIMD build.');
    }

    return null;
}

async function loadScaler() {
    const response = await fetch('./model/scaler.json');
    if (!response.ok) {
        throw new Error(`Could not download scaler.json (HTTP ${response.status} ${response.statusText})`);
    }
    return response.json();
}

async function init() {
    console.log(`Starting initialization (onnxruntime-web ${ORT_VERSION})...`);

    const unsupported = checkBrowserSupport();
    if (unsupported) {
        console.error('Initialization aborted:', unsupported.title);
        predictBtn.textContent = 'Unsupported Browser';
        showError(unsupported.title, unsupported.advice);
        return;
    }

    // If the CDN is unreachable the library itself never defines `ort`.
    if (typeof ort === 'undefined') {
        console.error('Initialization aborted: the onnxruntime-web library did not load.');
        predictBtn.textContent = 'Error Loading Model';
        showError(
            'The machine learning library could not be downloaded.',
            'ONNX Runtime is loaded from the jsDelivr CDN (cdn.jsdelivr.net), which may be ' +
            'blocked on this network or by a browser extension. Try a different network, ' +
            'or allow cdn.jsdelivr.net, then reload the page.'
        );
        return;
    }

    // Serve the WebAssembly binaries from the same pinned version as the library
    // in index.html. Mismatched versions fail to initialise.
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

    // GitHub Pages is not cross-origin isolated, so SharedArrayBuffer (and with
    // it multi-threading) is unavailable. Ask for a single thread up front to
    // avoid a needless warning and a threaded build that cannot be used.
    ort.env.wasm.numThreads = 1;

    // Load the scaler and the model separately so a failure points at the
    // step that actually broke rather than at a single catch-all.
    try {
        scalerParams = await loadScaler();
    } catch (e) {
        console.error('Initialization failed while loading the scaler:', e);
        predictBtn.textContent = 'Error Loading Model';
        showError(
            'Could not load the model parameters.',
            'The scaler file could not be downloaded. Check your network connection and reload the page.',
            e.message
        );
        return;
    }

    try {
        session = await ort.InferenceSession.create('./model/model.onnx');
    } catch (e) {
        console.error('Initialization failed while creating the inference session:', e);
        predictBtn.textContent = 'Error Loading Model';
        showError(
            'Could not start the machine learning runtime.',
            'The ONNX Runtime files are loaded from the jsDelivr CDN. This usually means the ' +
            'browser is too old, or the CDN is blocked on this network. If you are using Safari, ' +
            'update to 16.4 or newer, or try Chrome, Edge or Firefox.',
            e.message
        );
        return;
    }

    console.log('Model loaded successfully. Outputs:', session.outputNames);
    statusMessage.classList.add('hidden');
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict Toxicity';
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const inputs = [
        parseFloat(document.getElementById('R1_P1nv').value),
        parseFloat(document.getElementById('R2_P1nv').value),
        parseFloat(document.getElementById('R3_P1nv').value),
        parseFloat(document.getElementById('R4_P1nv').value),
        parseFloat(document.getElementById('R5_P1nv').value),
        parseFloat(document.getElementById('R6_P1nv').value),
        parseFloat(document.getElementById('R7_P1nv').value)
    ];

    const scaledInputs = inputs.map((val, i) => {
        return (val - scalerParams.mean[i]) / scalerParams.scale[i];
    });

    try {
        const data = Float32Array.from(scaledInputs);
        const tensor = new ort.Tensor('float32', data, [1, 7]);
        
        // Input name is usually 'X' or 'input'
        const feeds = { [session.inputNames[0]]: tensor };
        const results = await session.run(feeds);
        
        console.log("Inference results:", results);

        // With ZipMap disabled:
        // results.label -> tensor with predicted class [0 or 1]
        // results.probabilities -> tensor with probabilities [prob_class_0, prob_class_1]
        
        const label = Number(results.label.data[0]);
        const toxicProb = results.probabilities.data[1];

        // `label` uses the model's built-in 0.5 argmax, which we override with
        // TOXIC_THRESHOLD. Log when the two disagree so the divergence is
        // visible rather than silent.
        const thresholdSaysToxic = toxicProb >= TOXIC_THRESHOLD;
        if (thresholdSaysToxic !== (label === 1)) {
            console.warn(
                `Model label (${label === 1 ? 'Toxic' : 'Normal'}, 0.5 argmax) disagrees with ` +
                `the ${TOXIC_THRESHOLD} threshold (${thresholdSaysToxic ? 'Toxic' : 'Normal'}) ` +
                `at P(toxic)=${toxicProb.toFixed(2)}. Using the threshold.`
            );
        }

        console.log(`P(toxic)=${toxicProb.toFixed(2)} -> ${verdictFor(toxicProb).text}, ` +
                    `confidence ${Math.round(confidenceFor(toxicProb) * 100)}%`);
        displayResult(toxicProb);
    } catch (e) {
        console.error("Inference failed:", e);
        alert(`Prediction error: ${e.message}`);
    }
});

/**
 * How confident the model is in the reported class.
 *
 * Scales the distance between P(toxic) and the decision threshold onto
 * 50%-100%: a result sitting exactly on the threshold is a coin flip (50%),
 * and P(toxic) of 0 or 1 is complete confidence (100%). Confidence never drops
 * below 50%, because below that we would be reporting the other class.
 *
 * This is a distance from the decision boundary, NOT a calibrated probability:
 * "82% confidence" does not mean "82% chance of toxicity".
 */
function confidenceFor(rawToxicProb) {
    // Snap to the forest's true 1% resolution first. The model returns float32,
    // so an exact 0.22 arrives as 0.2199999839; without this, values landing on
    // a rounding tie can display inconsistently.
    const toxicProb = Math.round(rawToxicProb * 100) / 100;

    const distance = toxicProb < TOXIC_THRESHOLD
        ? (TOXIC_THRESHOLD - toxicProb) / TOXIC_THRESHOLD
        : (toxicProb - TOXIC_THRESHOLD) / (1 - TOXIC_THRESHOLD);
    return 0.5 + 0.5 * distance;
}

/**
 * Three-state verdict derived from P(toxic) alone.
 * Results close to the threshold are reported as indeterminate.
 */
function verdictFor(rawToxicProb) {
    // The model returns float32, so an exact 0.30 arrives as 0.30000001192.
    // Snap to the forest's true 1% resolution before comparing to the bounds.
    const toxicProb = Math.round(rawToxicProb * 100) / 100;

    if (toxicProb >= UNCERTAIN_LOWER && toxicProb <= UNCERTAIN_UPPER) {
        return {
            text: 'Uncertain',
            className: 'uncertain',
            note: 'Borderline result.'
        };
    }
    if (toxicProb > UNCERTAIN_UPPER) {
        return { text: 'Toxic', className: 'toxic', note: '' };
    }
    return { text: 'Normal', className: 'normal', note: '' };
}

function displayResult(toxicProb) {
    resultSection.classList.remove('hidden');

    const verdict = verdictFor(toxicProb);
    // The forest has 100 trees, so P(toxic) is quantised to whole percent;
    // showing more precision than this would be misleading.
    const confidence = Math.round(confidenceFor(toxicProb) * 100);

    predictionLabel.textContent = verdict.text;
    predictionLabel.className = verdict.className;

    resultDetails.textContent = `Model confidence: ${confidence}%`;
    if (verdict.note) {
        resultDetails.textContent += ` - ${verdict.note}`;
    }

    // The bar fills to the confidence value itself, so it reads directly
    // against the number shown above it.
    confidenceFill.style.width = `${confidence}%`;
    confidenceFill.className = 'confidence-fill bg-' + verdict.className;

    resultSection.scrollIntoView({ behavior: 'smooth' });
}

init();
