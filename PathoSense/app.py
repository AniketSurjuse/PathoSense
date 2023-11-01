from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model
import pickle

model = load_model('CNN_model.h5')
diabetes_model = pickle.load(open(r"Diabetes_prediction.pckl","rb"))
heart_disease_model = pickle.load(open(r"heart_disease_prediction.pckl",'rb'))

app = Flask(__name__)

# Function that simulates the 'predict' function
def pneumonia_predictinon(img):
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
    
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshaped =  input_data_as_numpy_array.reshape(1,-1)
    
    prediction = diabetes_model.predict(input_data_reshaped)
    
    if prediction[0]==1:
        return "diabetic"
    else:
        return "Normal"

def heart_disease_prediction(input_data):
    input_data_np_array = np.array(input_data)
    
    reshaped_input_data = input_data_np_array.reshape(1,-1)
    
    prediction = heart_disease_model.predict(reshaped_input_data)
    
    if prediction[0]==1:
        return "You might have heart disease"
    else:
        return "You do NOT have heart disease"

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        uploaded_image = request.files['image']

        if uploaded_image.filename != '':
            img = Image.open(uploaded_image)
            

            prediction_result = pneumonia_predictinon(img)
            data = prediction_result

        return render_template('result.html',data=data)
    return render_template('index.html')


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    # prediction_result = "default"
    if request.method == 'POST':
        input_fields = ["preg", "glucose", "bp", "skinthickness", "insulin", "bmi", "pedigree", "age"]
        
        # Collect form input values into a list
        input_list = []
        for i in input_fields:
            input_list.append(float(request.form.get(i)))


        # Call the prediction method with the input list
        prediction_result = diabetes_prediction(input_list)


        return render_template('result.html',data = prediction_result)
    
    
    return render_template('diabetes.html')

@app.route('/heartd', methods=['GET', 'POST'])
def heartd():
    if request.method == 'POST':
        # Collect form data
        input_fields = ['age','sex','cp','trestbps','chol','fbs','restcg','thalach','exang','oldpeak','slop','ca','thal']
        input_list = []
        for i in input_fields:
            val = request.form.get(i)
            if val == None:
                val =0
            input_list.append(float(val))

        predictions = heart_disease_prediction(input_list)

        return render_template('result.html',data = predictions)
    return render_template('heartd.html')







if __name__ == '__main__':
    app.run(debug=True)
