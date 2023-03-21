from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np



def statified_results(X, y, classifier, N=5):

    skf = StratifiedKFold(n_splits=N)
    skf.get_n_splits(X, y)

    f1_scores_classwise = [] 

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
            
        clf = classifier() 
        clf.fit(X_train, y_train)
        
        y_preds = clf.predict(X_test)

        _, _, f, s_ = precision_recall_fscore_support(y_test, y_preds)
        f1_scores_classwise.append(f.tolist())
    return  np.mean(f1_scores_classwise, axis=0), np.std(f1_scores_classwise, axis=0)



from sklearn.linear_model import LogisticRegression

def logistic_train():
  return LogisticRegression(solver='liblinear')


from sklearn.tree import DecisionTreeClassifier

def dt():
  return DecisionTreeClassifier()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda():
  return LinearDiscriminantAnalysis()


from sklearn.svm import SVC

def svm():
  return SVC(kernel='poly', degree=5)


from sklearn.ensemble import RandomForestClassifier

def rf():
  return RandomForestClassifier()


from sklearn.neighbors import KNeighborsClassifier

def knn():
    return KNeighborsClassifier()


import numpy as np

from catboost import CatBoostClassifier


def catboost():
    model = CatBoostClassifier(iterations=15,
                               depth=16,
                               learning_rate=0.2,
                               loss_function='MultiClass',
                               verbose=True)
    return model

def catboost2():
    model = CatBoostClassifier(iterations=25,
                               depth=16,
                               learning_rate=0.1,
                               loss_function='MultiClass',
                               verbose=True)
    return model



import lightgbm as ltb

def light():
    params = {
              "num_leaves": 150,
              "learning_rate": 0.3,
              "n_estimators": 220,
              "max_depth": 200,
              "class_weight": 'balanced'
             }
    
    model = ltb.LGBMClassifier(**params)
    return model
    


  


# @skopt.utils.use_named_args(SPACE)
# def objective(**params):
#     reg.set_params(**params)

#     return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    # scoring="neg_mean_absolute_error"))
# def objective(**params):
#    return -1.0 * train_evaluate(params)


#  @skopt.utils.use_named_args(SPACE)
#         def objective(**params):
#             # clf_sig.set_params(**params)
#             lbm_clf = ltb.LGBMClassifier(**params)
#             return -np.mean(cross_val_score(lbm_clf, X, y, cv=5, n_jobs=-1,
#                                             scoring="neg_mean_absolute_error"))
        
#         from skopt import gp_minimize

#         result = gp_minimize(objective, SPACE, n_calls=50, random_state=42, verbose=1)
        
        
#         results = skopt.forest_minimize(objective, SPACE,
#                                 n_calls=100, n_random_starts=10)
#         print("results light::", results)

