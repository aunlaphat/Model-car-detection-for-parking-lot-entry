import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

app = Flask(__name__)

# Load model
MODEL_PATH = 'model14_3_2.keras'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
model = load_model(MODEL_PATH)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load subclass mapping
def load_subclass_mapping():
    return {
        'EV': {
            0: 'BYD-Atto-3',
            1: 'BYD-Dolphin',
            2: 'BYD-sealPerformance',
            3: 'MG-4',
            4: 'MG-EP',
            5: 'MG-ZS-EVStandard',
            6: 'neta-v',
            7: 'ORA-Good-Cat',
            8: 'Tesla-3',
            9: 'Tesla-Y',
        },
        'Non-EV': {
            10: 'Ford-Ranger-Rapter-3.0',
            11: 'Honda-HR-V-e_HEV',
            12: 'Isuzu-d-max-hi-lander-1.3',
            13: 'Isuzu-Mu-x',
            14: 'Mitsubishi-Triton',
            15: 'Nissan-Almera',
            16: 'Toyota-Corolla-Cross',
            17: 'Toyota-Fortuner',
            18: 'Toyota-hilus-revo-prerunner',
            19: 'Toyota-yaris-ATIV',
        }
    }

# Function to prepare image
def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Prepare the image and make prediction
        img = prepare_image(file_path)
        predictions = model.predict(img)

        # Load subclass mapping
        subclass_mapping = load_subclass_mapping()

        ev_prediction = np.round(predictions[0])[0]  # EV/Non-EV
        subclass_prediction = np.argmax(predictions[1], axis=1)[0]  
        
        # Get probabilities for each subclass
        subclass_probabilities = predictions[1][0]  # Get the probabilities for the first image

        # Print values for debugging
        print(f"Predictions: EV prediction: {ev_prediction}, Subclass predictions: {subclass_prediction}")
        print(f"subclass_probabilities: {subclass_probabilities}")
        # Determine subclass name and probabilities
        if subclass_prediction in subclass_mapping['EV']:
            subclass_name = subclass_mapping['EV'][subclass_prediction]
            ev_status = 'EV'
        elif subclass_prediction in subclass_mapping['Non-EV']:
            subclass_name = subclass_mapping['Non-EV'][subclass_prediction]
            ev_status = 'Non-EV'
        else:
            subclass_name = "Unknown"
            ev_status = "Unknown"


        # Clear Keras session to free memory
        K.clear_session()

        # Return result to the index page
        return render_template('index.html', ev_status=ev_status, subclass_name=subclass_name, image_path=file_path)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
