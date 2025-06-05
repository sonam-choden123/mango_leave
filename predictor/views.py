import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Path to the model file
MODEL_PATH = os.path.join('mango.h5')
model = load_model(MODEL_PATH)

# Class labels
classes = ['Sooty Mould','Powdery Mildew','Healthy','Gall Midge','Die Back','Cutting Weevil','Bacterial Canker','Anthracnose']

# Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict_image(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return classes[predicted_class]

# View function
def index(request):
    if request.method == 'POST' and request.FILES.get('leaf_image'):
        try:
            file = request.FILES['leaf_image']
            path = default_storage.save('temp.jpg', file)
            image = Image.open(default_storage.path(path)).convert('RGB')
            prediction = predict_image(image)
            return render(request, 'predictor/predict.html', {'prediction': prediction})
        except Exception as e:
            return render(request, 'predictor/predict.html', {'error': str(e)})
    return render(request, 'predictor/predict.html')
