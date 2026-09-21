// Must match the onnxruntime-web version of the <script> tag in index.html.
const ORT_VERSION = '1.18.0';

let session;
let scalerParams;

const form = document.getElementById('prediction-form');
const predictBtn = document.getElementById('predict-btn');
const resultSection = document.getElementById('result-section');
const predictionLabel = document.getElementById('prediction-label');
const resultIndicator = document.getElementById('result-indicator');
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
        
        console.log(`Prediction: ${label === 1 ? 'Toxic' : 'Normal'}, Toxic Prob: ${toxicProb}`);
        displayResult(label);
    } catch (e) {
        console.error("Inference failed:", e);
        alert(`Prediction error: ${e.message}`);
    }
});

function displayResult(label) {
    resultSection.classList.remove('hidden');
    const isToxic = label === 1;
    predictionLabel.textContent = isToxic ? "Toxic" : "Normal";
    predictionLabel.className = isToxic ? "toxic" : "normal";
    resultIndicator.className = "indicator " + (isToxic ? "bg-toxic" : "bg-normal");
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

init();
