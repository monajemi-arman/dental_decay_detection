import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import io
import base64
from infer import infer
from waitress import serve
import nrrd
import tempfile

# Path to model onnx file
ckpt_path = '3.ckpt'

# Host and port for waitress server
host = '0.0.0.0'
port = 33517

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'nrrd'}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    image_data = None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        # if user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            # Support NRRD files
            if file.filename.rsplit('.', 1)[1].lower() == 'nrrd':
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(file.getvalue())
                    tmp.flush()
                    pixels, _ = nrrd.read(tmp.name)
                    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255
                    pixels = pixels.squeeze().astype(np.uint8)
                    img = Image.fromarray(pixels)
            else:
                # Convert to JPG if PNG
                img = Image.open(file)
                if img.format == 'PNG':
                    img = img.convert('RGB')
            output = io.BytesIO()
            img.save(output, format='JPEG')
            image_data = output.getvalue()
            # Encoding the image and showing in HTML (no files saved to disk)
            image_data_orig = base64.b64encode(image_data).decode('utf-8')

            image_data = np.array(img)
            image_data = infer(image_data, ckpt_path)
            img = Image.fromarray(image_data)
            # To byte in order to encode
            output = io.BytesIO()
            img.save(output, format='JPEG')
            image_data = output.getvalue()
            image_data = base64.b64encode(image_data).decode('utf-8')
            # Show
            return render_template('index.html', message='File uploaded successfully',
                                   image_data_orig=image_data_orig, image_data=image_data)

    return render_template('index.html')


if __name__ == '__main__':
    serve(app.wsgi_app, host=host, port=port)
