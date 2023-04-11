from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Beat these results
# [0.8333333333333333, 0.4, 0.6666666666666666, 0.6666666666666666]

def train_clf(X, y, clf):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(X) // 5)
  clf.fit(X_train, y_train)
  
  y_preds = clf.predict(X_test)

  _, _, f, s_ = precision_recall_fscore_support(y_test, y_preds)
  return f.tolist()

# From stratified_results
# (Pdb) len(y_train), len(y_test)
# (55, 14)


def plot_catboost_features(X_train, y_train, feat_names, clf, TOP = 50):
  import matplotlib.pyplot as plt
  from catboost import CatBoostClassifier, Pool
  train_data = Pool(data=X_train, label=y_train, cat_features=[])
  feat_importance = clf.get_feature_importance(train_data, )
  # feat_names = self.feat_names # clf.feature_names_
  print(feat_importance)
  fig, ax = plt.subplots(figsize=(8, 6))
  sorted_zipped = sorted(list(zip(feat_names, feat_importance)), key=lambda x: x[1], reverse=True)

  sorted_feat, sorted_imp =  zip(*sorted_zipped)
  ax.barh([str(x) for x in sorted_feat[:TOP]], sorted_imp[:TOP])
  ax.set_xlabel('Feature Importance')
  ax.set_title('CatBoost Classifier Feature Importance')
  plt.show()



def statified_results(X, y, project_names, classifier, self, N=5):

    skf = StratifiedKFold(n_splits=N)
    skf.get_n_splits(X, y)
    project_names = np.array(project_names)
    test_projects = None
    f1_scores_classwise = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        
        clf = classifier

        clf.fit(X_train, y_train)
        
        y_preds = clf.predict(X_test)

        _, _, f, s_ = precision_recall_fscore_support(y_test, y_preds)
        f1_scores_classwise.append(f.tolist())
        test_projects = project_names[test_index]

    plot_catboost_features(X_train, y_train, self.feat_names, clf)

    # breakpoint()
    # feature_importance()
    return  np.mean(f1_scores_classwise, axis=0), np.std(f1_scores_classwise, axis=0), test_projects, y_test, clf



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
                               learning_rate=0.1,
                               loss_function='MultiClass',
                               verbose=True)
    return model

def catboost2():
    model = CatBoostClassifier(iterations=25,
                               depth=16,
                               learning_rate=0.2,  # 0.1
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
    