from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# ----------------------
# ROUTES
# ----------------------

@app.route('/')
def home():
    """Landing page (grade predictor form)"""
    return render_template('index.html')


@app.route('/about')
def about():
    """Static About page added by user (about.html)"""
    return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Contact form – shows confirmation on successful POST"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Simple console log for now; replace with DB/email integration if needed
        print(f"Contact Form: {name} | {email} | {message}")

        # Pass success=True so the template can show a confirmation alert
        return render_template('contact.html', success=True)

    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle grade prediction requests"""
    if request.method == 'POST':
        form_data = request.form.to_dict()
        name = form_data.pop('naam', 'Student')  # Default name if none provided

        try:
            # Map form fields → model features
            class_year = float(form_data['class_year'])           # proxy for age or academic year
            studytime = float(form_data['studytime'])
            failures = min(float(form_data['failures']), 3)       # Cap failures at 3

            # Convert attendance percentage to absences out of 200 days
            attendance = float(form_data['attendance'])
            absences = 200 - ((attendance / 100) * 200)
            absences = max(0, min(absences, 200))

            G1 = float(form_data['G1']) * 0.2                     # 20% weight
            G2 = float(form_data['G2']) * 0.2                     # 20% weight

            features = [class_year, studytime, failures, absences, G1, G2]
            prediction = model.predict([features])[0]
            prediction = max(0, min(prediction, 20))              # Clamp between 0–20

            percentage = (prediction / 20) * 100

            # Convert percentage to letter grade
            if percentage >= 90:
                grade = "A+ (Excellent)"
            elif percentage >= 75:
                grade = "A (Very Good)"
            elif percentage >= 60:
                grade = "B (Good)"
            elif percentage >= 40:
                grade = "C (Needs Improvement)"
            else:
                grade = "F (Fail)"

            return render_template(
                'index.html',
                prediction_text=f'{name}, your predicted final grade is {round(percentage, 2)}% — {grade}'
            )
        except (ValueError, KeyError):
            # Handles missing keys or non‑numeric input
            return render_template('index.html', prediction_text="Invalid input. Please enter valid numbers.")


if __name__ == '__main__':
    app.run(debug=True)
