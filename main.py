import base64

import numpy as np
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from io import BytesIO
from hrvanalysis import get_time_domain_features
import torch
from xresnet1d import ecg_inference


app = Flask(__name__)

def sanitize_hrv_metrics(hrv_metrics):
    sanitized_metrics = {}
    for key, value in hrv_metrics.items():
        if isinstance(value, (np.int32, np.int64)):
            sanitized_metrics[key] = int(value)  # 将 numpy.int 转换为 int
        elif isinstance(value, (np.float32, np.float64)):
            sanitized_metrics[key] = float(value)  # 将 numpy.float 转换为 float
        else:
            sanitized_metrics[key] = value
    return sanitized_metrics

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # 读取文件
    try:
        data = np.loadtxt(file.stream, delimiter=",")
    except Exception as e:
        return jsonify({"error": str(e)})

    # 绘制ECG
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    plt.figure()
    plt.plot(data[0], linewidth=1.2)
    plt.grid(linestyle='--')
    img_io = BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')

    # hrv分析
    hrv_metrics = get_time_domain_features(data[0]);
    hrv_metrics = sanitize_hrv_metrics(hrv_metrics)

    # 模型推理
    input_ecg = torch.from_numpy(data).float().unsqueeze(0)
    lead, prediction_result = ecg_inference(input_ecg)

    result = {
        "prediction_result": prediction_result,
        "hrv_metrics": hrv_metrics,
        "image": img_base64
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
