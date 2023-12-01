from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def train_LR_model_gridsearch(X_train, y_train, seed, C_values, score_crit):
    lr_model=LogisticRegression(random_state=seed)
    lr_model = GridSearchCV(lr_model, 
                            param_grid={'C': C_values}, 
                            scoring=score_crit)
    lr_model.fit(X_train, y_train)
    
    return lr_model