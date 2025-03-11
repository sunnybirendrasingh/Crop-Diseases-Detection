from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pickle

app = Flask(__name__, template_folder="template", static_folder='static')
model = pickle.load(open('graphs_classifier.pkl', 'rb'))  # import model

def model_output(path):
    raw_img = image.load_img(path, target_size=(64, 64))
    raw_img = image.img_to_array(raw_img)
    raw_img = np.expand_dims(raw_img, axis=0)
    raw_img = raw_img / 255.0  # Normalizing the image data
    probabilities = model.predict(raw_img)[0]

    leafs = ['Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 
               'Pepper Bacterial spot', 'Pepper healthy', 'Potato Early blight', 
               'Potato Late blight', 'Potato healthy', 'Tomato Bacterial spot', 
               'Tomato Early blight', 'Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot',
             'Tomato Spider mites','Tomato Target Spot','Tomato healthy','Tomato mosaic virus']

    max_prob_index = np.argmax(probabilities)
    max_prob = probabilities[max_prob_index]

    if max_prob > 0.5:
        result = f"It's {leafs[max_prob_index]}"
    else:
        result = "Class not confidently detected"

    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded file temporarily
    temp_path = 'static/temp_image.jpg'
    file.save(temp_path)

    # Process the image and get the result
    result = model_output(temp_path)

    # Pass the image path and result to the template
    return render_template('index.html', output_image='temp_image.jpg', output_text=result)

if __name__ == '__main__':
    app.run(debug=True)
