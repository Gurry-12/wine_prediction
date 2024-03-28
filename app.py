
#import the requirements
from flask import Flask,render_template,json,jsonify,request
import pickle
import numpy as np
import requests

#initialize the app

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        fixedAcidity=float(request.form['fixedAcidity'])
        volatileAcidity=float(request.form['volatileAcidity'])
        citric_acid=float(request.form['citricAcid'])
        residual_sugar=float(request.form['residualSugar'])
        chlorides=float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['freeSulfurDioxide'])
        total_sulfur_dioxide=float(request.form['totalSulfurDioxide'])
        density=float(request.form['density'])
        ph=float(request.form['pH'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])

        #load the pickle file
        filename='models/model.pkl'
        loaded_model=pickle.load(open(filename,'rb'))
        data=np.array([[fixedAcidity,volatileAcidity,citric_acid,residual_sugar,
                        chlorides,free_sulfur_dioxide,
                        total_sulfur_dioxide,density,ph,sulphates,alcohol]])
        my_prediction=loaded_model.predict(data)
        #get the result template
        return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)