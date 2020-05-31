FROM python:3.7.6

ADD application.py input_columns.joblib requirements.txt revenue_prediction_final_pipeline.joblib /

RUN pip install -r requirements.txt

ENV USER_1=test_user
ENV PASSWORD_1=sharkyshark95

CMD [ "python", "./application.py" ]