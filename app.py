from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Add pickel
model = pickle.load(open("./model/diabetes-model-knn.pickle", 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for('index'))
    
    try:
        data = np.array([[
            float(request.form.get('num-pregnancies')),
            float(request.form.get('glucose')),
            float(request.form.get('BMI')),
            float(request.form.get('age'))
        ]])
        prediction = model.predict(data)
        return render_template('index.html', response = str(prediction[0]))

    except Exception as error:
        print(error)
        return render_template('index.html', response = "Error : " + str(error))

# if __name__ == "__main__":
#     app.run(debug=True)