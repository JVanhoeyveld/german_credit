from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_selected):
        self.features_selected=features_selected


    def fit(self, X, y=None):
        return self
    

    def transform(self, X):
        try:
            X = X[self.features_selected].copy()
        except:
            raise ValueError('Warning: the selected features do not appear in the data to transform')
        return X
    

class ScalerMinMax(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables


    def fit(self, X, y=None):
        #variables to transform
        if self.variables is None:
            #if undefined, derive from X data (only numerical features!)
            data_types = pd.DataFrame(X.dtypes, columns=['type'])
            data_types_numerical = data_types.loc[data_types['type']!='object',:]
            vars_to_scale = list(data_types_numerical.index)
        else:
            vars_to_scale = self.variables
        self.vars_to_scale = vars_to_scale

        #for each variable that should be scaled, store min and max values of input data X
        vars_summary = {}
        for var in vars_to_scale:
            vars_summary[var] = dict()
            vars_summary[var]['min'] = X[var].min()
            vars_summary[var]['max'] = X[var].max()
        self.vars_summary = vars_summary
        return self
    

    def transform(self, X):
        X = X.copy() #to not overwrite original input data

        #apply min max transformation (scaling)
        for var in self.vars_to_scale:
            X[var] = (X[var]-self.vars_summary[var]['min'])/(self.vars_summary[var]['max']-self.vars_summary[var]['min'])
            X[var] = np.clip(X[var], 0, 1)

        return X