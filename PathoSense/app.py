from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model
model = load_model('CNN_model.h5')

app = Flask(__name__)

# Function that simulates the 'predict' function
def predict(img):
    image = img.resize((256,256))
    image = image.convert('RGB')
    img_array = np.array(image)
    img = img_array/255.0
    original_shape = (256, 256, 3)
    new_shape = (1,) + original_shape

    # Create a numpy array with the new shape
    new_array = np.empty(new_shape, dtype=np.uint8)
    
    y_pred = model.predict(new_array)
    if y_pred[0][0]==1:
        return "normal"
    else:
        return "pneumonia"
    

# result = []

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/pneumonia')
def help():
    return render_template('index.html')


@app.route('/pneumonia', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file from the form
        uploaded_image = request.files['image']

        if uploaded_image.filename != '':
            # Read the image and convert it to a numpy array
            img = Image.open(uploaded_image)
            

            # Call the predict function with the image data
            prediction_result = predict(img)
            # result.append(prediction_result)
            data = prediction_result

    return render_template('index.html',data=data)


if __name__ == '__main__':
    app.run(debug=True)
