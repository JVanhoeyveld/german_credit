#check if we can use the app defined in main.py
#(normally you would use pytest to do this and have more extensive tests)

from main import app

from fastapi.testclient import TestClient
import pickle
import numpy as np

#setup a test client to test fastapi application, see:
#https://fastapi.tiangolo.com/tutorial/testing/
client = TestClient(app)

#load test data
with open("VERSION") as version_file:
    __version__ = version_file.read().strip()
version_file.close()
with open('development/results_train_models/results_train_{}.pkl'.format(__version__), 'rb') as file:
    results_train = pickle.load(file)
file.close()
X_test = results_train['data_split']['X_test']

#take the first 2 rows for testing
X_test = X_test.iloc[0:2,:].copy()
print(X_test)
#convert to list of dictionaries compatible with RawDataScheme
X_test = X_test.replace({np.nan: None}).to_dict(orient="records")
print(X_test)

#decide on payload that will be supplied to the api post request
#-->the same structure as you can see when trying out the example in the docs of the api /prediction/batch
payload = dict()
payload["inputs"] = X_test

#call the api to make predictions
response = client.post('/prediction/batch', json=payload)
print(response.status_code)
print(response.json())


#how to test GET endpoint with parameters...
responst_2 = client.get('/something_random_with_params',
                        params={'a': 4,
                                'b':5})
print(responst_2.json())