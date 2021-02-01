from flask import Flask, request, render_template,jsonify
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

f = open('store.pckl', 'rb')
to_store = pickle.load(f)
f.close()

encodings = to_store['encodings']
rfc = to_store['rfc']

app = Flask(__name__)

def performPrediction(name, age, workclass, edunum, marital, occupation, relationship, gender, hours, gain, loss) :

	workclass = encodings['workclass'][workclass]
	marital = encodings['marital.status'][marital]
	relationship = encodings['relationship'][relationship]
	occupation = encodings['occupation'][occupation]
	gender = encodings['sex'][gender]

	array = np.array([[age, workclass, edunum, occupation, marital, relationship, gender, gain, loss, hours]])

	num = rfc.predict(array)

	for key, value in encodings['income'].items() :
		if value == num[0] :
			num = key

	return num

@app.route('/')

def home() :
	return render_template('Dashboard.html')

@app.route('/join', methods=['GET','POST'])

def my_form_post() :
	name = request.form['name']
	age = request.form['age']
	workclass = request.form['workclass']
	edunum = request.form['edunum']
	marital = request.form['marital']
	occupation = request.form['occupation']
	relationship = request.form['relation']
	gender = request.form['gender']
	hours = request.form['hours']
	gain = request.form['gain']
	loss = request.form['loss']

	combine = performPrediction(name, age, workclass, edunum, marital, occupation, relationship, gender, hours, gain, loss)

	result = {"output": combine}
	result = {str(key): value for key, value in result.items()}

	return jsonify(result=result)

if __name__ == '__main__' :
	app.run(debug=True)