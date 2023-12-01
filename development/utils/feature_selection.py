import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def univariate_FS_LR(X_train, y_train, features_nominal, features_cont,
                     seed, C_values):
    #(very basic: fit a LR on a single variable and see result on AUC train)
    summary = pd.DataFrame(columns=['feature','auc_score'])
    for feat in features_nominal:
        x_feat = X_train[feat].copy()
        include_missing_dummy = False
        if x_feat.isna().sum() > 0:
            include_missing_dummy = True    
        x_feat = pd.get_dummies(x_feat, drop_first=True, dummy_na=include_missing_dummy)
        x_feat.columns = x_feat.columns.astype(str)
        lr_model=LogisticRegression(random_state=seed)
        lr_model = GridSearchCV(lr_model, 
                                param_grid={'C':C_values}, 
                                scoring='roc_auc')
        lr_model.fit(x_feat, y_train)
        new_row = pd.DataFrame({'feature':feat, 'auc_score':lr_model.best_score_}, index=[0])
        summary = pd.concat([summary.loc[:], new_row]).reset_index(drop=True)
    for feat in features_cont:
        x_feat = X_train[feat].copy() 
        x_feat.fillna(x_feat.mean(), inplace=True)
        x_feat= x_feat.values.reshape(-1, 1) #needed to fit lr model https://stackoverflow.com/questions/51150153/valueerror-expected-2d-array-got-1d-array-instead
        lr_model=LogisticRegression(random_state=seed)
        lr_model = GridSearchCV(lr_model, 
                                param_grid={'C':C_values}, 
                                scoring='roc_auc')
        lr_model.fit(x_feat, y_train)
        new_row = pd.DataFrame({'feature':feat, 'auc_score':lr_model.best_score_}, index=[0])
        summary = pd.concat([summary.loc[:], new_row]).reset_index(drop=True)
    summary.sort_values(by='auc_score', ascending=False, inplace=True)
    feat_selected = summary.iloc[0:5,0]
    feat_selected = list(feat_selected.values)
    return feat_selected