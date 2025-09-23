import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from infer import Predictor
from waitress import serve
import os
from werkzeug.datastructures import FileStorage # Added import for FileStorage

# Host and port for waitress server
host = '0.0.0.0'
port = 33517

app = Flask(__name__)

# Enable CORS and allow all hosts
CORS(app, resources={r"/*": {"origins": "*"}})

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'nrrd'}

predictor = Predictor()

# --- New: Demo mode setup ---
script_dir = os.path.dirname(os.path.realpath(__file__))
PREDICTIONS_JSON_PATH = os.path.join(script_dir, 'predictions.json')
PREDICTIONS_JSON_EXISTS = os.path.exists(PREDICTIONS_JSON_PATH)
PREDICTIONS_DATA_MAPPED = {} # Renamed to avoid confusion with raw COCO data
if PREDICTIONS_JSON_EXISTS:
    try:
        with open(PREDICTIONS_JSON_PATH, 'r') as f:
            raw_coco_predictions = json.load(f)

        # Create a mapping from image_id to normalized filename
        image_id_to_filename = {
            img['id']: img['file_name'].lower()
            for img in raw_coco_predictions.get('images', [])
        }

        # Group annotations by normalized filename
        for ann in raw_coco_predictions.get('annotations', []):
            image_id = ann.get('image_id')
            if image_id in image_id_to_filename:
                filename = image_id_to_filename[image_id]
                if filename not in PREDICTIONS_DATA_MAPPED:
                    PREDICTIONS_DATA_MAPPED[filename] = []
                # Store the annotation directly, as draw_boxes expects bbox and category_id
                PREDICTIONS_DATA_MAPPED[filename].append(ann)

        print(f"Loaded predictions from {PREDICTIONS_JSON_PATH} and mapped to filenames.")
        # print(f"DEBUG: Mapped predictions keys: {PREDICTIONS_DATA_MAPPED.keys()}") # Optional: for debugging
    except Exception as e:
        print(f"Error loading predictions.json: {e}")
        PREDICTIONS_JSON_EXISTS = False
# --- End New ---

# --- New: Demo folder setup ---
DEMO_FOLDER_NAME = 'demo'
DEMO_FOLDER_PATH = os.path.join(script_dir, DEMO_FOLDER_NAME)
DEMO_FOLDER_EXISTS = os.path.exists(DEMO_FOLDER_PATH) and os.path.isdir(DEMO_FOLDER_PATH)

def load_demo_images():
    demo_images_list = []
    if DEMO_FOLDER_EXISTS:
        # Sort files for consistent display order
        for filename in sorted(os.listdir(DEMO_FOLDER_PATH)):
            if allowed_file(filename):
                filepath = os.path.join(DEMO_FOLDER_PATH, filename)
                try:
                    img = Image.open(filepath)
                    img = img.convert('RGB')

                    # Create a thumbnail
                    thumbnail_size = (200, 150) # Define thumbnail size (width, height)
                    img.thumbnail(thumbnail_size, Image.LANCZOS) # Use LANCZOS for better quality resizing

                    output = io.BytesIO()
                    img.save(output, format='JPEG')
                    base64_data = base64.b64encode(output.getvalue()).decode('utf-8')
                    demo_images_list.append({
                        'filename': filename,
                        'base64_data': base64_data
                    })
                except Exception as e:
                    print(f"Error loading demo image {filename}: {e}")
    return demo_images_list
# --- End New ---

def create_mock_file_storage(filepath, filename, mimetype='image/jpeg'):
    """Creates a FileStorage-like object from a local file path."""
    with open(filepath, 'rb') as f:
        file_content = io.BytesIO(f.read())
    # FileStorage expects a stream, filename, name (form field name), and mimetype
    # We'll use 'file' as the form field name to match the existing logic
    return FileStorage(stream=file_content, filename=filename, name='file', content_type=mimetype)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_boxes(image, annotations):
    draw = ImageDraw.Draw(image)
    line_width = max(1, int(min(image.width, image.height) * 0.005))

    font_size = int(min(image.width, image.height) * 0.03)  # 3% of smaller dimension
    font = None

    try_fonts = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans.ttf"]
    for font_name in try_fonts:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except IOError:
            continue

    if font is None:
        font = ImageFont.load_default()
        print("Warning: Using non-scalable default font. For better results, install a TTF font.")

    for ann in annotations:
        # Expecting 'bbox' as [x, y, width, height] and 'category_id'
        x, y, width, height = ann['bbox']

        draw.rectangle([x, y, x + width, y + height], outline="red", width=line_width)

        # label = str(int(ann['category_id']))

        # text_bbox = draw.textbbox((0, 0), label, font=font)
        # text_width = text_bbox[2] - text_bbox[0]
        # text_height = text_bbox[3] - text_bbox[1]

        # text_x = x
        # text_y = y - text_height - 2  # 2px padding
        # if text_y < 0:
        #     text_y = y  # Move inside if at top edge

        # # Draw label background
        # draw.rectangle(
        #     [text_x, text_y, text_x + text_width, text_y + text_height],
        #     fill="red"
        # )

        # # Draw label text
        # draw.text((text_x, text_y), label, fill="white", font=font)

    return image


# Modified process_image to accept demo_predictions
def process_image(file, gt_file=None, gt_json=None, infer_model=True, demo_predictions=None):
    global predictor

    img = Image.open(file)
    img = img.convert('RGB')

    output = io.BytesIO()
    img.save(output, format='JPEG')
    image_data_orig = base64.b64encode(output.getvalue()).decode('utf-8')

    annotations = []
    gt_data = None
    if gt_json is not None:
        image_id = None
        # Use file.filename directly for matching, assuming gt_json images have 'file_name'
        for image_info in gt_json.get('images', []):
            if image_info.get('file_name') == file.filename:
                image_id = image_info.get('id')
                break
        if image_id is not None:
            annotations = [ann for ann in gt_json.get('annotations', []) if ann.get('image_id') == image_id]
    elif gt_file and gt_file.filename.endswith('.json'):
        # This block is for single GT file upload, which is not used in the multi-file scenario with gt_json
        # but kept for completeness if single file upload path is used.
        gt_json_local = json.load(gt_file)
        image_id = None
        for image_info in gt_json_local.get('images', []):
            if image_info.get('file_name') == file.filename:
                image_id = image_info.get('id')
                break
        if image_id is not None:
            annotations = [ann for ann in gt_json_local.get('annotations', []) if ann.get('image_id') == image_id]

    if annotations:
        img_gt = draw_boxes(img.copy(), annotations)
        gt_output = io.BytesIO()
        img_gt.save(gt_output, format='JPEG')
        gt_data = base64.b64encode(gt_output.getvalue()).decode('utf-8')

    inferred_data = None
    predictions = []

    if not infer_model and demo_predictions:
        print(f"DEBUG: process_image for {file.filename}: Using demo predictions. Count: {len(demo_predictions)}") # Debug print
        # Ensure demo_predictions have a 'score' key for the template
        processed_demo_predictions = []
        for ann in demo_predictions:
            processed_ann = ann.copy() # Create a copy to avoid modifying original loaded data
            if 'score' not in processed_ann:
                processed_ann['score'] = -1.0 # Default score if missing
            processed_demo_predictions.append(processed_ann)

        predictions = processed_demo_predictions
        img_inferred = draw_boxes(img.copy(), demo_predictions)  # Draw demo boxes on original image
        inferred_output = io.BytesIO()
        img_inferred.save(inferred_output, format='JPEG')
        inferred_data = base64.b64encode(inferred_output.getvalue()).decode('utf-8')
    elif infer_model:
        print(f"DEBUG: process_image for {file.filename}: Performing model inference.") # Debug print
        # Perform actual model inference
        image_array = np.array(img)
        inferred_array, preds = predictor.infer(image_array, details=True)
        predictions = preds  # preds from infer.py are already in the desired format for frontend
        img_inferred = Image.fromarray(inferred_array)
        inferred_output = io.BytesIO()
        img_inferred.save(inferred_output, format='JPEG')
        inferred_data = base64.b64encode(inferred_output.getvalue()).decode('utf-8')

    return {
        "original_image": image_data_orig,
        "ground_truth_image": gt_data,
        "inferred_image": inferred_data,
        "predictions": predictions
    }


@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({"status": "success", "message": "API is running"}), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    files = request.files.getlist('file')
    if len(files) == 0:
        return jsonify({"status": "error", "message": "No selected file"}), 400

    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Invalid file format"}), 400

    # This endpoint does not support demo mode or multiple files with GT JSON in the same way as the frontend.
    # It will always run inference if infer_model=True (default).
    # For simplicity, keeping this endpoint as is, focusing demo mode on the frontend.
    if len(files) == 1:
        file = files[0]
        gt_file = request.files.get('gt_file')
        try:
            result = process_image(file, gt_file=gt_file, infer_model=True)  # Always infer for API
            response = {
                "status": "success",
                "message": "Inference successful",
                "data": {
                    "original_image": result['original_image'],
                    "ground_truth_image": result['ground_truth_image'],
                    "inferred_image": result['inferred_image'],
                    "predictions": result['predictions']
                }
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        # This part of the API endpoint is for multiple files, but it doesn't handle demo_predictions
        # and assumes inference.
        gt_file = request.files.get('gt_file')
        gt_json = None
        if gt_file:
            if not gt_file.filename.endswith('.json'):
                return jsonify({"status": "error", "message": "GT file must be a JSON"}), 400
            try:
                gt_json = json.load(gt_file)
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error loading GT JSON: {str(e)}"}), 400

        results = []
        for file in files:
            try:
                result = process_image(file, gt_json=gt_json, infer_model=True)  # Always infer for API
                results.append({
                    "filename": file.filename,
                    "predictions": result['predictions']
                })
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error processing {file.filename}: {str(e)}"}), 500

        response = {
            "status": "success",
            "message": "Inference successful for multiple files",
            "data": results
        }
        return jsonify(response), 200


# Ground Truth Visualization Endpoint
@app.route('/api/v1/ground-truth', methods=['POST'])
def ground_truth():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    gt_file = request.files.get('gt_file')

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            result = process_image(file, gt_file, infer_model=False)  # Never infer for GT endpoint
            response = {
                "status": "success",
                "message": "Ground truth processed successfully",
                "data": {
                    "original_image": result['original_image'],
                    "ground_truth_image": result['ground_truth_image']
                }
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Invalid file format"}), 400


# Front-end Endpoint
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Load demo images regardless of GET or POST, so they are always available
    # if the folder exists.
    demo_images = load_demo_images()

    if request.method == 'POST':
        files_to_process = []
        gt_file_for_processing = None
        demo_mode_for_processing = False
        message = 'Files uploaded successfully' # Default message

        # Check if a demo image was chosen
        chosen_demo_filename = request.form.get('chosen_demo_filename')

        if chosen_demo_filename:
            # A demo image was chosen via the "Choose" button
            demo_filepath = os.path.join(DEMO_FOLDER_PATH, chosen_demo_filename)
            if os.path.exists(demo_filepath) and allowed_file(chosen_demo_filename):
                mock_file = create_mock_file_storage(demo_filepath, chosen_demo_filename)
                files_to_process.append(mock_file)
                demo_mode_for_processing = request.form.get('demo_mode_for_demo') == 'on'
                print(f"DEBUG: Chosen demo image: {chosen_demo_filename}. Demo mode: {demo_mode_for_processing}")
            else:
                message = f"Error: Demo image '{chosen_demo_filename}' not found or invalid."
                # Render template with error and existing demo images
                return render_template('index.html',
                                       message=message,
                                       predictions_json_exists=PREDICTIONS_JSON_EXISTS,
                                       demo_mode_checked=request.form.get('demo_mode_for_demo') == 'on',
                                       demo_images=demo_images)
        else:
            # Regular file upload from the main form
            files_to_process = request.files.getlist('file')
            gt_file_for_processing = request.files.get('gt_file')
            demo_mode_for_processing = request.form.get('demo_mode') == 'on'
            print(f"DEBUG: Regular upload. Demo mode: {demo_mode_for_processing}")

        if not files_to_process or all(f.filename == '' for f in files_to_process):
            return render_template('index.html',
                                   message='No selected file(s)',
                                   predictions_json_exists=PREDICTIONS_JSON_EXISTS,
                                   demo_mode_checked=demo_mode_for_processing,
                                   demo_images=demo_images)

        # Check if all files are allowed
        for f in files_to_process:
            if not allowed_file(f.filename):
                return render_template('index.html',
                                       message=f'Invalid file format for {f.filename}',
                                       predictions_json_exists=PREDICTIONS_JSON_EXISTS,
                                       demo_mode_checked=demo_mode_for_processing,
                                       demo_images=demo_images)

        gt_json = None
        if gt_file_for_processing and gt_file_for_processing.filename.endswith('.json'):
            try:
                gt_json = json.load(gt_file_for_processing)
            except Exception as e:
                return render_template('index.html',
                                       message=f"Error loading GT JSON: {str(e)}",
                                       predictions_json_exists=PREDICTIONS_JSON_EXISTS,
                                       demo_mode_checked=demo_mode_for_processing,
                                       demo_images=demo_images)

        all_results = []
        for file in files_to_process:
            if file and allowed_file(file.filename):
                try:
                    current_demo_predictions = None
                    should_infer_model = True

                    # Normalize filename for lookup
                    normalized_filename = file.filename.lower()
                    print(f"DEBUG: Processing file: {file.filename}, Normalized filename: {normalized_filename}")

                    # Check if demo mode is active and predictions exist for this file
                    if demo_mode_for_processing and PREDICTIONS_JSON_EXISTS and normalized_filename in PREDICTIONS_DATA_MAPPED:
                        current_demo_predictions = PREDICTIONS_DATA_MAPPED[normalized_filename]
                        should_infer_model = False
                        print(f"DEBUG: Demo mode active and predictions found for {normalized_filename}. Infer model: {should_infer_model}. Predictions count: {len(current_demo_predictions) if current_demo_predictions else 0}")
                    else:
                        print(f"DEBUG: Demo mode not active or no predictions found for {normalized_filename} in PREDICTIONS_DATA_MAPPED. Infer model: {should_infer_model}")


                    result = process_image(file, gt_json=gt_json, infer_model=should_infer_model, demo_predictions=current_demo_predictions)
                    all_results.append({
                        "filename": file.filename,
                        "original_image": result['original_image'],
                        "ground_truth_image": result['ground_truth_image'],
                        "inferred_image": result['inferred_image'],
                        "predictions": result['predictions']
                    })
                except Exception as e:
                    all_results.append({
                        "filename": file.filename,
                        "error": f"Error processing {file.filename}: {str(e)}"
                    })
            else:
                all_results.append({
                    "filename": file.filename,
                    "error": f"Invalid file format for {file.filename}"
                })

        return render_template(
            'index.html',
            message=message,
            all_results=all_results,
            predictions_json_exists=PREDICTIONS_JSON_EXISTS,
            demo_mode_checked=demo_mode_for_processing,
            demo_images=demo_images
        )

    else: # GET request
        return render_template('index.html',
                               predictions_json_exists=PREDICTIONS_JSON_EXISTS,
                               demo_mode_checked=PREDICTIONS_JSON_EXISTS, # Default checked if JSON exists
                               demo_images=demo_images)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("                 WEB APPLICATION IS RUNNING!")
    print("                 ===========================")
    print(f"\n        Please open your web browser and navigate to:")
    print(f"        \033[1m\033[96mhttp://localhost:{port}\033[0m")
    print("\n" + "=" * 80 + "\n")
    serve(app.wsgi_app, host=host, port=port)
