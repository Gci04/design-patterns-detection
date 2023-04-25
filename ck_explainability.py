import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import shap
from catboost import CatBoostClassifier
from sklearn.inspection import PartialDependenceDisplay
from time import time
import matplotlib.pyplot as plt

import faulthandler

from sklearn.model_selection import StratifiedKFold




def statified_results(X, y, classifier, feat_cols, N=5):

    skf = StratifiedKFold(n_splits=N)
    skf.get_n_splits(X, y)
    # project_names = np.array(project_names)
    test_projects = None
    f1_scores_classwise = []

    all_fols_shaps = []
    all_folds_clfs = []
    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        clf = classifier

        clf.fit(X_train, y_train)
        
        y_preds = clf.predict(X_test)

        all_folds_clfs.append(clf)

        _, _, f, s_ = precision_recall_fscore_support(y_test, y_preds)
        f1_scores_classwise.append(f.tolist())
        print("F1 curr split: ", f)
        shaps_1fold = apply_shapely(clf, X_train, y_train, feat_cols)
        all_fols_shaps.append(shaps_1fold)
        # test_projects = project_names[test_index]
    print("catboost features plots...")
    
    mean_shaps_classwise = np.mean(np.array(all_fols_shaps), axis=0)

    plot_features(mean_shaps_classwise, feat_cols)
    return  np.mean(f1_scores_classwise, axis=0), np.std(f1_scores_classwise, axis=0)  # , test_projects, y_test, clf


def plot_features(mean_shaps_classwise, features):

    TOPN_FEATS = 50
    id2class = {0:'MVC', 1:'MVP', 2: 'MVVM', 3: 'NONE'}
    color = ['green','blue','maroon','orange']
    for i in range(len(color)):
        shap = mean_shaps_classwise[i]
        idx = np.argsort(shap).tolist()
        #print("Index", idx)
        shap = shap[idx][::-1]
        shap = shap[:TOPN_FEATS]
        #print("SHAP", shap)
        feats = np.array(features)[idx][:TOPN_FEATS]
        plt.barh(feats,shap,color=color[i])
        plt.ylabel("Features")
        plt.xlabel("Mean(|SHAP values|)")
        plt.title(f"SHAP values For Class {id2class[i]} | TOP: {TOPN_FEATS}")
        plt.tick_params(axis='y', which='major', labelsize=5)

        plt.tight_layout()

        plt.savefig(f'SHAP values for class {id2class[i]}')
        plt.show()



##CATBOOST

def catboost():
    model = CatBoostClassifier(iterations=15,
                               depth=16,
                               learning_rate=0.1,
                               loss_function='MultiClass',
                               verbose=True)
    return model



def apply_shapely(model, x_train, y_train, features):
    # Debug segmentation fault
    faulthandler.enable()

    model.fit(x_train,y_train)

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(x_train)

    return get_shap(shap_values)


def get_shap(shap_values):
    color = ['green','blue','maroon','orange']

    all_shaply_vals = [np.mean(np.abs(shap_values[i]), axis=0) for i in range(len(color))]
    return all_shaply_vals


def get_shap_deprecated(shap_values, features, id2class):
    color = ['green','blue','maroon','orange']
    for i in range(len(color)):
        feats = features
        shap = np.mean(np.abs(shap_values[i]),axis=0)
        idx = np.argsort(shap).tolist()
        #print("Index", idx)
        shap = shap[idx][::-1]
        shap = shap[:35]
        #print("SHAP", shap)
        feats = np.array(features)[idx][:35]
        plt.barh(feats,shap,color=color[i])
        plt.ylabel("Features")
        plt.xlabel("Mean(|SHAP values|)")
        plt.title(f"SHAP values For Class {id2class[i]}")
        plt.tick_params(axis='y', which='major', labelsize=5)

        plt.tight_layout()

        plt.savefig(f'SHAP values for class {id2class[i]}')
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv("ck_metrics_embeddings.csv")
    #print(data.columns.tolist()
    X, y = data.loc[:,data.columns != 'label'], data['label']
    # x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=42)
    print(statified_results(X.values, y, catboost(), X.columns.tolist()))