import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request

app = Flask(__name__)

with open('svc_trained_model_03.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    chest_pain = int(request.form['chest_pain'])
    max_heart_rate = int(request.form['max_heart_rate'])
    input_data = [[age, gender, chest_pain, max_heart_rate]]
    input_data = StandardScaler().fit_transform(input_data)
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        result = 'Have Disease'
    else:
        result = 'Not Have Disease'
    return render_template('index.html', result=result, age = age, sex=gender, cp=chest_pain, thalach=max_heart_rate)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
