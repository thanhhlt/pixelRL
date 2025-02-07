from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
import threading
import time
from denoise.main import denoise_image

app = Flask(__name__)

active_tasks = {}

def process_image(task, image, task_id):
    if task == "denoise":
        return denoise_image(image, task_id, active_tasks)
    return None

@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files or "task" not in request.form:
        return jsonify({"error": "Missing file or task"}), 400

    file = request.files["file"]
    task = request.form["task"]

    img_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    task_id = str(time.time())
    active_tasks[task_id] = {"status": "processing", "thread": None, "image": None}

    def process_task():
        processed_image = process_image(task, image, task_id)

        if processed_image is None:
            active_tasks[task_id]["status"] = "failed"
            return

        _, buffer = cv2.imencode(".jpg", processed_image)
        active_tasks[task_id]["image"] = buffer.tobytes()
        active_tasks[task_id]["status"] = "done"

    thread = threading.Thread(target=process_task)
    active_tasks[task_id]["thread"] = thread
    thread.start()

    return jsonify({"task_id": task_id})

@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    if task_id not in active_tasks:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({"task_id": task_id, "status": active_tasks[task_id]["status"]})

@app.route("/result/<task_id>", methods=["GET"])
def get_result(task_id):
    if task_id not in active_tasks:
        return jsonify({"error": "Task not found"}), 404

    if active_tasks[task_id]["status"] != "done":
        return jsonify({"error": "Task not finished"}), 400

    return send_file(io.BytesIO(active_tasks[task_id]["image"]), mimetype="image/jpeg")

@app.route("/cancel/<task_id>", methods=["DELETE"])
def cancel_task(task_id):
    if task_id in active_tasks and active_tasks[task_id]["status"] == "processing":
        active_tasks[task_id]["status"] = "canceled"
        return jsonify({"status": "canceled"})
    return jsonify({"error": "Task not found or already finished"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)