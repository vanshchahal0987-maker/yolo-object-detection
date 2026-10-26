from flask import Flask, request, render_template
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

model = YOLO("yolov8n.pt")


@app.route("/", methods=["GET", "POST"])
def home():
    result_image = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            error = "Please select an image first."
        else:
            filename = secure_filename(file.filename)
            input_path = os.path.join(
                app.config["UPLOAD_FOLDER"], filename
            )
            file.save(input_path)

            results = model(input_path)
            result_filename = "detected_" + filename
            result_path = os.path.join(
                app.config["RESULT_FOLDER"], result_filename
            )

            results[0].save(filename=result_path)
            result_image = "/" + result_path.replace("\\", "/")

    return render_template(
        "index.html",
        result_image=result_image,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)