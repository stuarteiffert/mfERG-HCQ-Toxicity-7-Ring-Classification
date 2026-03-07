let session;
let scalerParams;

const form = document.getElementById('prediction-form');
const predictBtn = document.getElementById('predict-btn');
const resultSection = document.getElementById('result-section');
const predictionLabel = document.getElementById('prediction-label');
const resultIndicator = document.getElementById('result-indicator');

// Configure ONNX Runtime to use CDN for Wasm files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

async function init() {
    console.log("Starting initialization...");
    try {
        const scalerResponse = await fetch('./model/scaler.json');
        if (!scalerResponse.ok) throw new Error(`Failed to fetch scaler.json: ${scalerResponse.statusText}`);
        scalerParams = await scalerResponse.json();

        session = await ort.InferenceSession.create('./model/model.onnx');
        console.log("Model loaded successfully. Outputs:", session.outputNames);
        
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict Toxicity';
    } catch (e) {
        console.error("Initialization failed:", e);
        predictBtn.textContent = 'Error Loading Model';
    }
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
