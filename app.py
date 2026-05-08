from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load dataset
data = pd.read_csv("student_data.csv")

X = data[['study_hours', 'attendance', 'previous_marks']]
y = data['final_score']

# Load model
model = joblib.load("model.pkl")

# Accuracy
y_pred = model.predict(X)
accuracy = r2_score(y, y_pred)

# Graph
hours = np.linspace(1, 10, 10)
attendance = 80
previous_marks = 70

predictions = []

for h in hours:
    pred = model.predict([[h, attendance, previous_marks]])
    predictions.append(pred[0])

plt.figure()
plt.plot(hours, predictions)
plt.xlabel("Study Hours")
plt.ylabel("Predicted Score")
plt.title("Study Hours vs Predicted Score")
plt.savefig("static/graph.png")
plt.close()

# Login page
@app.route('/', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        # Simple login
        if username == "student" and password == "1234":
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid Username or Password")

    return render_template('login.html')

# Predictor page
@app.route('/home', methods=['GET', 'POST'])
def home():

    prediction = None

    if request.method == 'POST':

        study_hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        previous_marks = float(request.form['previous'])

        features = np.array([[study_hours, attendance, previous_marks]])

        prediction = model.predict(features)[0]

    return render_template(
        'index.html',
        prediction=prediction,
        accuracy=round(accuracy * 100, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)