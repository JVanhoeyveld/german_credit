from deployment.schemas.data_schema import RawDataScheme
from development.predict import predict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd

app = FastAPI()


#the home page
@app.get("/", response_class=HTMLResponse)
def welcome():
    content="""
    <html>
        <body>
            <h1>Welcome to the German Credit API</h1>
                <p>Check the docs <a href='/docs'>here</a></p>
        </body>
    </html>
    """
    return content


#To try out get endpoint with parameters (see test_main.py how to test it)
@app.get('/something_random_with_params')
def test(a: float = 0, b: float = 0):
    return {'product': a*b,
            'sum':a+b}


@app.post('/prediction/batch')
def prediction(input_data: RawDataScheme):
    #convert to pandas df
    input_data = pd.DataFrame(jsonable_encoder(input_data.inputs)) #list of pydantic obj --> dict --> pandas DF
    #predict for the input data (see development folder)
    (prediction_binary, prediction_prob) = predict(input_data)

    results={'binary predictions': [int(pred) for pred in prediction_binary],
             'probability predictions':[float(pred) for pred in prediction_prob]}

    return results