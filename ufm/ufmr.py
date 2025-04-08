import os
import sys
import socket
import logging
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as T
import base64

# ✅ Block stdout – required by Claude MCP
sys.stdout = open(os.devnull, 'w')

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Import internal components (make sure they don't print to stdout)
from agent import FederatedUFMSystem
from utils import plot_patch_overlay_on_image

# ✅ Auto-pick open port to avoid crashes
def get_open_port(start=5057, end=5060):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError("No open port found")

# ✅ Claude MCP `initialize` endpoint (required)
@app.route('/', methods=['POST'])
def initialize():
    try:
        # Step 1.1: Log the content type
        print(f"📡 Content-Type: {request.content_type}", file=sys.stderr)

        # Step 1.2: Check if content type is correct
        if request.content_type != 'application/json':
            print(f"❌ Invalid content type", file=sys.stderr)
            return jsonify({"error": "Invalid content type"}), 400

        # Step 1.3: Force parse JSON and log it
        data = request.get_json(force=True)
        print(f"📥 Received Init Payload: {data}", file=sys.stderr)

        # Step 1.4: Check if it's a valid MCP 'initialize' request
        if data and data.get("method") == "initialize":
            print("✅ Valid MCP initialize request", file=sys.stderr)
            return jsonify({
                "jsonrpc": "2.0",
                "id": data["id"],
                "result": {
                    "serverInfo": { 
                        "name": "ufm-server",
                        "version": "0.1.0"
                    },
                    "capabilities": {}
                }
            })
        else:
            print("❌ Missing or invalid method", file=sys.stderr)
            return jsonify({"error": "Invalid method"}), 400

    except Exception as e:
        print(f"❌ MCP Init Exception: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


# ✅ Main Federated Diagnosis endpoint
@app.route('/functions/diagnose', methods=['POST'])
def diagnose():
    print("✅ /functions/diagnose endpoint hit", file=sys.stderr)
    try:
        patient_id = request.form['patient_id'].strip()
        age = int(request.form['age'])
        bp = int(request.form['bp'])
        hr = int(request.form['hr'])
        report = request.form['report']
        xray_file = request.files['xray_image']
        print(f"📥 Received: ID={patient_id}, Age={age}, BP={bp}, HR={hr}", file=sys.stderr)

        try:
            img = Image.open(xray_file.stream).convert("RGB")
        except Exception as e:
            print(f"❌ Invalid image: {e}", file=sys.stderr)
            return jsonify({"error": "Uploaded file is not a valid image"}), 400

        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)
        tab_tensor = torch.tensor([[age, bp, hr]], dtype=torch.float32)

        fed = FederatedUFMSystem(num_agents=3)
        results = fed.run_all(tab_tensor, report, img_tensor, patient_id)

        decisions, confidences = zip(*[(r[1], r[2]) for r in results])
        majority_label = max(set(decisions), key=decisions.count)
        majority_diagnosis = "Pneumonia" if majority_label.lower() != "normal" else "Normal"

        agent_1 = fed.agents[0]
        memory = agent_1.memory[patient_id]

        overlay_path = plot_patch_overlay_on_image(memory["img_contribs"], 10, 10, img_tensor)
        with open(overlay_path, "rb") as f:
            encoded_overlay = base64.b64encode(f.read()).decode("utf-8")

        tokenizer = agent_1.tokenizer
        token_ids = tokenizer(report, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        scores = memory["attn"][0].detach().cpu().numpy().flatten()
        top_tokens = sorted([
            (tokens[i], float(scores[i])) for i in range(min(len(tokens), len(scores))) if tokens[i] != "<pad>"
        ], key=lambda x: x[1], reverse=True)[:10]

        tab_contribs = [
            float(val.flatten()[0].item() if val.numel() > 1 else val.item())
            for val in memory["tab_contribs"]
        ]

        print("✅ Diagnosis complete, sending result", file=sys.stderr)
        return jsonify({
            "majority_diagnosis": majority_diagnosis,
            "agent_predictions": [
                {"agent": r[0], "label": r[1], "probability": round(r[2], 4)} for r in results
            ],
            "agent_1_diagnosis": {
                "label": "Pneumonia" if confidences[0] > 0.5 else "Normal",
                "confidence": round(confidences[0], 4),
                "top_text_tokens": top_tokens,
                "tabular_contributions": {
                    "Age": round(tab_contribs[0], 4),
                    "BP": round(tab_contribs[1], 4),
                    "HR": round(tab_contribs[2], 4)
                },
                "patch_overlay_base64": encoded_overlay
            }
        })

    except Exception as e:
        print(f"❌ Diagnosis error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

# ✅ Start the server
if __name__ == "__main__":
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    port = get_open_port()
    print(f"🚀 Federated UFM Flask server running on port {port}", file=sys.stderr)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# import os
# import sys
# import logging
# import traceback
# from flask import Flask, request, jsonify

# # Block stdout for Claude
# sys.stdout = open(os.devnull, 'w')

# # Setup Flask app
# app = Flask(__name__)

# @app.route('/', methods=['POST'])
# def initialize():
#     try:
#         print("🟢 /initialize POST hit", file=sys.stderr)
#         if request.content_type != 'application/json':
#             print(f"❌ Invalid content type: {request.content_type}", file=sys.stderr)
#             return jsonify({"error": "Invalid content type"}), 400

#         data = request.get_json(force=True)
#         print(f"📥 Init Payload: {data}", file=sys.stderr)

#         if data.get("method") == "initialize":
#             print("✅ Valid MCP initialize", file=sys.stderr)
#             return jsonify({
#                 "jsonrpc": "2.0",
#                 "id": data.get("id", 0),
#                 "result": {
#                     "serverInfo": {
#                         "name": "ufm-server",
#                         "version": "0.1.0"
#                     },
#                     "capabilities": {}
#                 }
#             })
#         else:
#             return jsonify({"error": "Invalid method"}), 400

#     except Exception as e:
#         print(f"❌ Init error: {e}", file=sys.stderr)
#         traceback.print_exc(file=sys.stderr)
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     logging.getLogger('werkzeug').setLevel(logging.ERROR)
#     port = int(os.environ.get("PORT", 5057))
#     print(f"🚀 UFM server running on port {port}", file=sys.stderr)
#     try:
#         app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
#     except Exception as e:
#         print(f"❌ Server crashed: {e}", file=sys.stderr)
#         traceback.print_exc(file=sys.stderr)
