from .schemas.data_schema import RawDataScheme

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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


from development.predict import predict
from fastapi.encoders import jsonable_encoder

@app.post('/prediction/batch')
def prediction(input_data: RawDataScheme):
    #convert to pandas df
    input_data = pd.DataFrame(jsonable_encoder(input_data.inputs)) #list of pydantic obj --> dict --> pandas DF
    #predict for the input data (see development folder)
    (prediction_binary, prediction_prob) = predict(input_data)

    results={'binary predictions': [int(pred) for pred in prediction_binary],
             'probability predictions':[float(pred) for pred in prediction_prob]}

    return results

