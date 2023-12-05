import pandas as pd
import pickle

#load results of running train.py 
with open("VERSION") as version_file:
    __version__ = version_file.read().strip()

with open('development/results_train_models/results_train_{}.pkl'.format(__version__), 'rb') as file:
    results_train = pickle.load(file)
    file.close()
preprocessing_pipeline = results_train['preprocessing_pipeline']
lr_model = results_train['lr_model']


def predict(input_data: pd.DataFrame):
    #transform the input data with the pipeline
    input_data_prep = preprocessing_pipeline.transform(input_data)

    #obtain predictions
    prediction_binary = lr_model.predict(input_data_prep)
    prediction_prob = lr_model.predict_proba(input_data_prep)[:,1]

    return (prediction_binary, prediction_prob)


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    pd.pandas.set_option('display.max_columns', None)
    X_test = results_train['data_split']['X_test'] 
    y_test = results_train['data_split']['y_test']
    (prediction_binary, prediction_prob) = predict(X_test)
    result = pd.DataFrame({'actual':y_test,
                           'predicted_binary':prediction_binary,
                           'predicted_prob':prediction_prob})
    auc = roc_auc_score(y_test, prediction_prob)
    print(X_test)
    print(result) 
    print('the auc on test data is: {}'.format(auc))

