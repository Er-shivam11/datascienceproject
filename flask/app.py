import os
from flask import Flask, request, redirect, flash, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO('yolov8n.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            # Perform object detection on the uploaded image
            results = model(image)
            # Draw bounding boxes on the image
            for result in results.pred:
                box = result.xywh
                x1, y1, w, h = int(box[0]-box[2]/2), int(box[1]-box[3]/2), int(box[2]), int(box[3])
                x2, y2 = x1 + w, y1 + h
                label = result.names[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Save the image with the bounding boxes
            marked_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'marked_' + filename)
            cv2.imwrite(marked_img_path, image)
            # Render the results template with the detection results
            return render_template('result.html', results=results, filename=filename, marked_img=marked_img_path)
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/uploads/marked_<filename>')
def marked_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'marked_' + filename)


if __name__ == '__main__':
    app.run()
