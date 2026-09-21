"""Lower the ONNX IR/opset version of model.onnx for older browser runtimes.

skl2onnx 1.20 emits IR version 10 / opset 22, which onnxruntime-web 1.18.0
refuses to load ("Unsupported model IR version"). We pin the web app to 1.18.0
because it is the last release that ships a non-SIMD WebAssembly build and so
still runs on Safari <= 16.3.

The exported graph contains a single ai.onnx.ml TreeEnsembleClassifier node and
no default-domain operators, so lowering the declared versions does not change
any computation. The script verifies this by asserting that predictions are
unchanged before overwriting the model.

Usage:
    python downgrade_model.py
"""

import json

import numpy as np
import onnx
import onnxruntime as rt

MODEL_PATH = "model/model.onnx"
SCALER_PATH = "model/scaler.json"

# onnxruntime-web 1.18.0 supports IR version <= 9; 8 is comfortably inside that.
TARGET_IR_VERSION = 8
TARGET_OPSETS = [("ai.onnx.ml", 1), ("", 12)]

# Representative inputs spanning normal and toxic responses.
TEST_CASES = [
    [32.9, 19.7, 11.8, 8.2, 6.3, 5.6, 5.4],
    [10.0, 6.0, 4.0, 3.0, 2.0, 2.0, 2.0],
    [60.0, 40.0, 25.0, 16.0, 12.0, 10.0, 9.0],
    [5.0, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0],
]


def predict(model_bytes, scaler):
    session = rt.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    results = []
    for raw in TEST_CASES:
        scaled = [
            (value - scaler["mean"][i]) / scaler["scale"][i]
            for i, value in enumerate(raw)
        ]
        x = np.array([scaled], dtype=np.float32)
        label, probabilities = session.run(None, {input_name: x})
        results.append((int(label[0]), [round(float(p), 9) for p in probabilities[0]]))
    return results


def main():
    with open(SCALER_PATH) as handle:
        scaler = json.load(handle)

    model = onnx.load(MODEL_PATH)
    print(f"current: ir_version={model.ir_version} "
          f"opsets={[(o.domain, o.version) for o in model.opset_import]}")

    default_domain_nodes = [
        node.op_type for node in model.graph.node if node.domain in ("", "ai.onnx")
    ]
    if default_domain_nodes:
        raise SystemExit(
            "Refusing to downgrade: graph uses default-domain operators "
            f"{default_domain_nodes}, so lowering the opset may change behaviour."
        )

    before = predict(model.SerializeToString(), scaler)

    downgraded = onnx.helper.make_model(
        model.graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )
    del downgraded.opset_import[:]
    downgraded.opset_import.extend(
        onnx.helper.make_opsetid(domain, version) for domain, version in TARGET_OPSETS
    )
    downgraded.ir_version = TARGET_IR_VERSION
    onnx.checker.check_model(downgraded)

    after = predict(downgraded.SerializeToString(), scaler)
    if before != after:
        raise SystemExit(f"Predictions changed!\n  before={before}\n  after ={after}")

    onnx.save(downgraded, MODEL_PATH)
    print(f"written: ir_version={downgraded.ir_version} "
          f"opsets={[(o.domain, o.version) for o in downgraded.opset_import]}")
    print(f"verified {len(TEST_CASES)} predictions unchanged: {after}")


if __name__ == "__main__":
    main()
