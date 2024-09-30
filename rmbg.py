from flask import Flask, request, send_file, render_template_string
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import os
import zipfile
import base64

app = Flask(__name__)

# Load the ONNX model
session = ort.InferenceSession('model.onnx')

@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Background Removal App</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1 { color: #333; }
                #imagePreview { margin-top: 20px; }
                #imagePreview img { max-width: 200px; margin: 10px; }
            </style>
        </head>
        <body>
            <h1>Upload images to remove background</h1>
            <form action="/remove-background" method="post" enctype="multipart/form-data">
                <input type="file" name="images" multiple onchange="previewImages(event)">
                <input type="submit" value="Upload and Process">
            </form>
            <div id="imagePreview"></div>

            <script>
                function previewImages(event) {
                    var preview = document.getElementById('imagePreview');
                    preview.innerHTML = '';
                    var files = event.target.files;

                    for (var i = 0; i < files.length; i++) {
                        var file = files[i];
                        var reader = new FileReader();
                        
                        reader.onload = (function(file) {
                            return function(e) {
                                var img = document.createElement('img');
                                img.src = e.target.result;
                                preview.appendChild(img);
                            };
                        })(file);
                        
                        reader.readAsDataURL(file);
                    }
                }
            </script>
        </body>
        </html>
    ''')

def process_image(image):
    # Resize the image to the expected dimensions (e.g., 1024x1024)
    target_size = (1024, 1024)
    image = image.resize(target_size, Image.LANCZOS)

    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)  # Change to CHW format
    img_array = img_array[None, :]  # Add batch dimension

    # Run inference
    inputs = {session.get_inputs()[0].name: img_array}
    outputs = session.run(None, inputs)
    
    # Handle the output based on its shape
    result = outputs[0].squeeze()
    if len(result.shape) == 2:
        # If the output is a single channel (alpha mask)
        mask = (result * 255).astype(np.uint8)
        # Apply the mask to the original image
        image = np.array(image)
        image = np.concatenate([image, mask[:,:,np.newaxis]], axis=2)
        result_image = Image.fromarray(image, mode='RGBA')
    elif len(result.shape) == 3:
        # If the output is already an RGB image with background removed
        result = (result * 255).astype(np.uint8)
        result_image = Image.fromarray(result, mode='RGB')
    else:
        raise ValueError("Unexpected output format from the model")

    return result_image

@app.route('/remove-background', methods=['POST'])
def remove_background():
    if 'images' not in request.files:
        return "No file part", 400

    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return "No selected file", 400

    processed_images = []
    for file in files:
        try:
            image = Image.open(file.stream).convert('RGB')
            result_image = process_image(image)
            
            # Convert to base64 for preview
            buffered = io.BytesIO()
            result_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            processed_images.append((file.filename, img_str))
        except Exception as e:
            return f"Error processing {file.filename}: {str(e)}", 500

    # Render result page with image previews
    return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Processed Images</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1 { color: #333; }
                .image-container { margin-bottom: 20px; }
                .image-container img { max-width: 300px; }
            </style>
        </head>
        <body>
            <h1>Processed Images</h1>
            {% for filename, img_data in images %}
            <div class="image-container">
                <h3>{{ filename }}</h3>
                <img src="data:image/png;base64,{{ img_data }}" alt="{{ filename }}">
                <br>
                <a href="data:image/png;base64,{{ img_data }}" download="{{ filename }}_processed.png">Download</a>
            </div>
            {% endfor %}
        </body>
        </html>
    ''', images=processed_images)

if __name__ == '__main__':
    app.run(debug=True)
