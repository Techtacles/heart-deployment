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
    #return render_template("index.html",prediction_text="The probability of heart failure is   {}".format(pred))
    if pred >0.5:
    	return render_template("index.html",prediction_text=f"Your probability of heart failure is  {pred}.You have {pred*100}% chance of heart failure .This is high. You can cut back rigorous exercises,reduce your intake of protein to normalize your serum creatinine value.Take more multi-vitamins to create steady blood flow i.e. to increase ejection fraction and level of blood platelets in your blood.By these, you can help reduce your probability of heart failure.  ")
    else:
    	return render_template("index.html",prediction_text=f" The probability of heart failure is  {pred}. You have {pred*100}.You've got nothing to worry about. That's a normal range.You can regularize your protein intake and increase your intake of multi-vitamins to make your chances of heart failure even lower. ")
if __name__ == '__main__':
    app.run(debug=True)