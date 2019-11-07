import lightgbm as lgbm
from sklearn.multiclass import OneVsRestClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier, LGBMRegressor

def LGBM_Models(xtrain_tfidf,xvalid_tfidf,ytrain):
    clf_LGBM = OneVsRestClassifier(lgbm.LGBMClassifier(objective = 'multiclass', num_class = 41 ))
    clf_LGBM.fit(xtrain_tfidf, ytrain)
    predicted_LGBM = clf_LGBM.predict(xvalid_tfidf)
    return predicted_LGBM
