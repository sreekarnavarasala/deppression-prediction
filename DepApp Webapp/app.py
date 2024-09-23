# app.py

# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np

# # Load the trained model
# model_path = 'model.pkl'
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract data from form
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
    
#     # Make prediction
#     prediction = model.predict(final_features)
#     output = 'Placed' if prediction[0] == 1 else 'Not Placed'

#     return render_template('index.html', prediction_text='Prediction: {}'.format(output))

# if __name__ == "__main__":
#     app.run(debug=True)



###################################################################################################################


from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled pipeline model
with open('pipeline_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict')
def predict():
    return render_template('/predict.html')


@app.route('/make_predict', methods=['POST'])
def make_predict():
    prediction = None

    if request.method == 'POST':
        # Retrieve the form data from the request as raw input values
        age = int(request.form.get('age'))
        income = int(request.form.get('income'))
        marital_status = request.form.get('marital-status')
        smoking_status = request.form.get('smoking-status')
        children = int(request.form.get('children'))
        physical_activity = request.form.get('physical-activity')
        employment_status = request.form.get('employment-status')
        alcohol_consumption = request.form.get('alcohol-consumption')
        dietary_habits = request.form.get('dietary-habits')
        sleep_patterns = request.form.get('sleep-patterns')
        mental_illness = request.form.get('mental-illness')
        substance_abuse = request.form.get('substance-abuse')

        # Prepare the data for prediction (raw inputs, no conversion)
        input_data = [[age, marital_status, children, smoking_status, physical_activity, employment_status, income,     
                      alcohol_consumption, dietary_habits, sleep_patterns, mental_illness, substance_abuse]]

        # Get the prediction
        prediction = model.predict(input_data)[0]
        if prediction == 0:
            prediction = "No Depression"
        else:
            prediction = "Suffering from Depression"
        # print(prediction)

    return render_template('/predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
