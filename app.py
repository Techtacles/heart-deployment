from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
#import os
#from pathlib import Path
app = Flask(__name__)
#dire='C:\\Users\\USER\\Desktop\\NEW DEPLOYMENT\\model.pkl'

model3=pickle.load(open("model3.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model3.predict_proba(final)
    pred=prediction.reshape(2,)[1]
   #output=round(prediction[0],2)
    return render_template("index.html",prediction_text="The probability of heart failure is   {}".format(pred))


if __name__ == '__main__':
    app.run(debug=True)