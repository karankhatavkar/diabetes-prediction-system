# Diabetes Prediction Using K-Nearest Neighbors (KNN) Classification

#### https://diabetes-prediction-app.onrender.com

**Approximate Loading Time: 60 Seconds**

## Project Overview

This project aims to predict the likelihood of diabetes in individuals using the K-Nearest Neighbors (KNN) algorithm. The model is trained on a dataset from the Pima Indians Diabetes Database and deployed via a Flask web application.

## Table of Contents

-   [Diabetes Prediction Using K-Nearest Neighbors (KNN) Classification](#diabetes-prediction-using-k-nearest-neighbors-knn-classification) - [https://diabetes-prediction-app.onrender.com](#httpsdiabetes-prediction-apponrendercom)
    -   [Project Overview](#project-overview)
    -   [Table of Contents](#table-of-contents)
    -   [Dataset](#dataset)
    -   [Model Training](#model-training)
        -   [1. Importing Libraries](#1-importing-libraries)
        -   [2. Reading and Exploring the Data](#2-reading-and-exploring-the-data)
        -   [3. Data Preprocessing](#3-data-preprocessing)
        -   [4. Model Training and Evaluation](#4-model-training-and-evaluation)
        -   [Exporting the Model](#exporting-the-model)
    -   [Flask Application](#flask-application)
        -   [Structure](#structure)
        -   [app.py](#apppy)
    -   [Setup and Installation](#setup-and-installation)
        -   [Prerequisites](#prerequisites)
        -   [Installation Steps](#installation-steps)
        -   [1. Clone the repository:](#1-clone-the-repository)
        -   [2. Navigate to the project directory:](#2-navigate-to-the-project-directory)
        -   [3. Create a virtual environment and activate it:](#3-create-a-virtual-environment-and-activate-it)
        -   [4. Install the required packages:](#4-install-the-required-packages)
    -   [Usage](#usage)
        -   [1. Run the Flask application:](#1-run-the-flask-application)
        -   [2. Open your web browser and go to http://127.0.0.1:5000/](#2-open-your-web-browser-and-go-to-http1270015000)
        -   [3. Enter the patient data and click on the "Predict" button to see the result.](#3-enter-the-patient-data-and-click-on-the-predict-button-to-see-the-result)
    -   [Results](#results)
    -   [Contributing](#contributing)
    -   [License](#license)

## Dataset

The dataset used for this project is the Pima Indians Diabetes Database, which is available on Kaggle. The dataset contains several medical predictor variables and one target variable, `Outcome`.

Dataset - [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Model Training

### 1. Importing Libraries

We start by importing the necessary libraries for data manipulation, visualization, and model training.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import pickle
```

### 2. Reading and Exploring the Data

The dataset is read into a pandas DataFrame and basic exploratory data analysis is performed.

```python
data = pd.read_csv('diabetes.csv')
data.head()
```

### 3. Data Preprocessing

The data is preprocessed by handling missing values, scaling features, and encoding categorical variables if necessary.

```python
# Replace zeros with NaN and then replace NaN with column mean
col_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
for col in col_clean:
    data[col] = data[col].replace(0, np.NaN)
    data[col].fillna(data[col].mean(), inplace=True)
```

### 4. Model Training and Evaluation

A K-Nearest Neighbors (KNN) classifier is trained on the data and evaluated.

```python
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Exporting the Model

The trained model is saved using pickle for later use in the Flask application.

```python
pickle.dump(knn, open('model.pkl', 'wb'))
```

## Flask Application

The Flask application provides a web interface to interact with the diabetes prediction model.

### Structure

**app.py:** Contains the Flask application code.  
**model/:** Directory where the trained model is stored.  
**templates/:** Contains `index.html` for the front-end.  
**static/:** Contains resources for the application.

### app.py

The Flask application loads the trained model and sets up routes for the web interface.

```python
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array([[float(request.form.get('num-pregnancies')),
                      float(request.form.get('glucose')),
                      float(request.form.get('BMI')),
                      float(request.form.get('age'))]])
    prediction = model.predict(data)
    return render_template('index.html', response=str(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
```

## Setup and Installation

### Prerequisites

-   Python 3.6 or higher
-   pip (Python package installer)

### Installation Steps

### 1. Clone the repository:

```bash
git clone https://github.com/karankhatavkar/diabetes-prediction.git
```

### 2. Navigate to the project directory:

```bash
cd diabetes-prediction
```

### 3. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Flask application:

```bash
python app.py
```

### 2. Open your web browser and go to http://127.0.0.1:5000/

### 3. Enter the patient data and click on the "Predict" button to see the result.

## Results

The model achieves good accuracy on the test dataset:

-   Accuracy: 0.74
-   Precision: 0.68 for diabetic, 0.76 for non-diabetic
-   Recall: 0.88 for diabetic, 0.48 for non-diabetic
-   F1-score: 0.81 for diabetic, 0.57 for non-diabetic

## Contributing

Contributions are welcome! Please create a pull request or submit an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.
