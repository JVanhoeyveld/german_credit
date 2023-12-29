from deployment.schemas.data_schema import RawDataScheme
from development.predict import predict

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
import pandas as pd
from typing_extensions import Annotated

app = FastAPI()

templates = Jinja2Templates(directory="deployment/templates")

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

#extra (can be ignored for project):example of rendering a simple html template
@app.get("/hello/", response_class=HTMLResponse)
def hello(request:Request, name: str):
   return templates.TemplateResponse("hello.html", context={"request":request, "name":name})


#extra (can be ignored for the project): To try out get endpoint with parameters (see test_main.py how to test it)
@app.post('/something_random_with_params')
def test(a: float = 0, b: float = 0):
    return {'product': a*b,
            'sum':a+b}


#users of the app can provide JSON input data representing several observations with raw input features
# and the api will return output predictions in JSON format.
@app.post('/api/predict')
def prediction(input_data: RawDataScheme):
    #convert to pandas df
    input_data = pd.DataFrame(jsonable_encoder(input_data.inputs)) #list of pydantic obj --> dict --> pandas DF
    #predict for the input data (see development folder)
    (prediction_binary, prediction_prob) = predict(input_data)

    results={'binary predictions': [int(pred) for pred in prediction_binary],
             'probability predictions':[float(pred) for pred in prediction_prob]}

    return results


#user can submit a form (a post request) holding values of the raw input features for a single observation
#the html webpage can show the prediction results of our trained models in the same webpage (/predict route)
@app.post('/predict', response_class=HTMLResponse)
def prediction(request: Request, 
            astatus: Annotated[str,Form()] = None, 
            duration: Annotated[float,Form()] = None,
            chistory: Annotated[str,Form()] = None,
            purpose: Annotated[str,Form()] = None,
            housing: Annotated[str,Form()] = None):
    
    #make pandas dataframe of input features supplied by the user 
    #(user fills in the fields of html form and presses 'make prediction' button)
    input_data = pd.DataFrame(data={'Astatus':[astatus],
                                    'Duration':[duration],
                                    'CHistory':[chistory],
                                    'Purpose': [purpose],
                                    'Housing':[housing]})
    
    #predict for the input data (single observation)
    (prediction_binary, prediction_prob) = predict(input_data)
    prediction_binary = prediction_binary[0]
    prediction_prob = prediction_prob[0]

    #render the html template
    return templates.TemplateResponse("form.html", 
            context={'request':request, 'prediction_binary': prediction_binary,
                     'prediction_prob': prediction_prob})


#used when accessing the /predict route (no post request but show form)
@app.get('/predict', response_class=HTMLResponse)
def prediction(request: Request):
    return templates.TemplateResponse("form.html", context={'request':request})
