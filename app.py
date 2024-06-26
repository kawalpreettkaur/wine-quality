from flask import Flask, render_template, request, Blueprint
import numpy as np
from waitress import serve
import os
from utils import predictQuality

app=Flask(__name__)
site = Blueprint("kawal", __name__, "templates")

@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict" , methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [float_features]
    prediction = predictQuality(features)
    if(prediction[0]==1):
        output = "Good Quality"
    else:
        output = "Bad Quality"
    return render_template("index.html", prediction_text = "The Red Wine is of {} ".format(output))

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=os.environ.get('PORT', 5000))
