from flask import Flask, request, jsonify, abort
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from joblib import load
import pandas as pd
import os
import dotenv

dotenv.load_dotenv(".env", verbose=True)
final_pipeline = load('revenue_prediction_final_pipeline.joblib')
expected_columns = load('input_columns.joblib')

app = Flask(__name__)
auth = HTTPBasicAuth()
CORS(app)

users = {
    os.getenv('USER_1'): generate_password_hash(os.getenv('PASSWORD_1'))
}


@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

def check_validity(expected_cols, given_cols):
    intersection = [value for value in expected_cols if value in given_cols]
    if len(expected_cols) == len(intersection):
        return True
    else:
        return False


@app.route("/")
@auth.login_required
def hello():
    return "Hello, %s! Please send a JSON with a data instance to the route: predict" % auth.username()


@auth.login_required
@app.route('/predict', methods=['GET'])
def predict_revenue():
    if not request.json:
        abort(400, {'message': 'No JSON body received.'})

    data = pd.DataFrame.from_records([request.json])

    if check_validity(expected_cols=expected_columns, given_cols=data.columns.tolist()):
        pred_class = final_pipeline.predict(data)[0]
        pred_proba = round(final_pipeline.predict_proba(data)[0][1], 4)
        return jsonify({'class': str(pred_class), 'probability': str(pred_proba)}), 201
    else:
        abort(400, {'message': 'Bad input. Columns expected: %s' % str(expected_columns)})