#pipeline to preprocess the data
from sklearn.pipeline import Pipeline
from development.utils.preprocessing import SelectFeatures, ScalerMinMax
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import OneHotEncoder

def pipeline_preprocessing(X_train, features_selected, features_nominal, features_cont):#steps of the pipeline
    preprocessing_pipeline = Pipeline([
        # ===== SELECT FEATURES (if applicable) =====
        ('feature_selection', SelectFeatures(features_selected=features_selected)),

        # ===== MISSING VALUES (nominal attributes) =====
        # nominal attributes are imputed by the most frequent value
        ('missing_vals_nominal', CategoricalImputer(imputation_method='frequent', 
                                    variables=[var for var in features_nominal if var in features_selected])),

        #continuous attributes are imputed by the median value
        ('missing_vals_cont', MeanMedianImputer(imputation_method='median', 
                                    variables=[var for var in features_cont if var in features_selected])),

        # ===== DUMMY ENCODING =====
        ('dummy_encoding', OneHotEncoder(drop_last=True, 
                                        variables=[var for var in features_nominal if var in features_selected])),

        # ===== NORMALIZATION =====
        ('scaler', ScalerMinMax())

    ])
    #fit the pipeline to the training data of the fold
    preprocessing_pipeline.fit(X_train)

    return preprocessing_pipeline