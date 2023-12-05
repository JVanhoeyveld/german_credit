"""
goal is to build a model (pickle) for the german credit dataset.
Steps included are:
    *load data
    *split train/test
    *FS
    *preprocess
    *train model
"""
#imports
import pandas as pd
import numpy as np
pd.pandas.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
import pickle

from development.utils.feature_selection import univariate_FS_LR
from development.utils.LR_model import train_LR_model_gridsearch
from pipeline import pipeline_preprocessing

#constants
ID_col = 'ClientID'
target_col = 'Status'
features_nominal = ['Astatus','CHistory','Purpose','Saccount','Etime','Pstatus',
                    'Debtors','Property','Iplans','Housing','Job','Phone','Fworker']
features_cont = ['Duration','Camount','IRate','Residence','Age','Ncredits','Depend']


#parameters
test_fraction = 0.3
seed = 123
nr_features_used = 5
C_values = 2.0**np.arange(-10,5)
score_crit = 'roc_auc'


#load data
data_raw = pd.read_csv('development/raw_data/GermanCredit.csv')


#train test split
X_train, X_test, y_train, y_test = train_test_split(data_raw[[col for col in data_raw.columns if (col != ID_col and col !=target_col)]], 
                                                    data_raw[target_col], 
                                                    test_size=test_fraction, 
                                                    random_state=seed)


#feature selection
features_selected = univariate_FS_LR(X_train, y_train, features_nominal, features_cont,
                     seed, C_values)


#preprocess the data
preprocessing_pipeline = pipeline_preprocessing(X_train, features_selected, features_nominal, features_cont)
X_train_prep = preprocessing_pipeline.transform(X_train)
#X_test_prep = preprocessing_pipeline.transform(X_test) # if one wants to transform test data


#train final LR model
lr_model = train_LR_model_gridsearch(X_train_prep, y_train, seed, C_values, score_crit)


#persist everything
with open("VERSION") as version_file:
    __version__ = version_file.read().strip()

results_train = dict()
results_train['data_split'] = dict()
results_train['data_split']['X_train'] = X_train
results_train['data_split']['X_test'] = X_test
results_train['data_split']['y_train'] = y_train
results_train['data_split']['y_test'] = y_test
results_train['features_selected'] = features_selected
results_train['preprocessing_pipeline'] = preprocessing_pipeline
results_train['lr_model'] = lr_model

with open('development/results_train_models/results_train_{}.pkl'.format(__version__), 'wb') as file:
    pickle.dump(results_train, file)
    file.close()






