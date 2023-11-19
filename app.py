import os
import cv2
import numpy as np 
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)


# Load the pre-trained model
model = load_model('dhanvantari.h5')

# Define allowed extensions for image uploads
ALLOWED_EXTENSIONS = ['JPG', 'jpeg', 'png','jpg','JPEG','PNG']

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_plant_leaf(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        return False  # Image couldn't be loaded
    
    # Convert the image to the HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define a green color range in HSV
    lower_green = np.array([35, 50, 50])  # Lower bound for green in HSV
    upper_green = np.array([90, 255, 255])  # Upper bound for green in HSV
    
    # Create a mask to extract green regions
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    
    # Calculate the percentage of green pixels in the image
    green_pixel_percentage = (np.count_nonzero(mask) / mask.size) * 100
    
    # Set a threshold for green pixel percentage (adjust as needed)
    green_threshold = 2  # For example, require at least 5% of green pixels
    
    # Find contours in the green mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if the image has enough green pixels to resemble a plant leaf
    # and if it contains at least one contour (leaf-like structure)
    return green_pixel_percentage >= green_threshold and len(contours) > 0


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload1.html')

@app.route('/upload', methods=['POST'])
def upload_detect():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        img_path = os.path.join('uploads', filename)
        
        # Perform a check to ensure the uploaded image is a valid plant leaf image
        if is_valid_plant_leaf(img_path):
            return detect_disease(img_path)
        else:
            print('invalid image')
            return render_template('invalid.html')
    else:
        return redirect(request.url)

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/camera/live')
def camera_live():
    return render_template('camera_live.html')

@app.route('/camera', methods=['POST'])
def camera_detect():
    data = request.get_json()
    image_data_url = data.get('imageDataURL')
    print(image_data_url)

    if image_data_url:
        # Decode the base64 image data to bytes
        image_data_bytes = base64.b64decode(image_data_url.split(',')[1])

        # Save the image data to a file
        with open('uploads/camera.jpg', 'wb') as f:
            f.write(image_data_bytes)

        img_path = 'uploads/camera.jpg'
        print(img_path)

        if is_valid_plant_leaf(img_path):
            return detect_disease(img_path)
        else:
            return render_template('invalid.html')
    else:
        return jsonify({'error': 'Image data URL not provided'})

def detect_disease(img_path,):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # Use your trained model for prediction here
    prediction = model.predict(img)
    disease_class = np.argmax(prediction)

    result = open("C:\\Users\\saibalaji\\Desktop\\Deploy_ml\\labels.txt", 'r').readlines()
    treatment =open("C:\\Users\\saibalaji\\Desktop\\Deploy_ml\\solutions.txt", 'r').readlines()
    class_name=result[disease_class]
    sol=treatment[disease_class]
    confidence_score = prediction[0][disease_class] * 100.0  # Convert to percentage
    formatted_confidence = "{:.2f}".format(confidence_score)
    print(class_name[2:])
    print(formatted_confidence)
    print(sol)
    # Return the class name and confidence score to the template
    return render_template('result.html',result=class_name[2:], confident=formatted_confidence, treatment=sol[2:])


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
    


