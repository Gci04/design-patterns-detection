import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lime
from lime import lime_tabular
from interpret.blackbox import LimeTabular, MorrisSensitivity
from interpret import show
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





data = pd.read_csv("embeddings.csv")
#print(data.columns.tolist()
X, y = data.loc[:,data.columns != 'label'], data['label']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=42)




##CATBOOST

model = CatBoostClassifier(iterations=15,
                               depth=16,
                               learning_rate=0.1,
                               loss_function='MultiClass',
                               verbose=True)


#print(y_train)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
#print(y_train)
#print(model.classes_)
##Adding RandomForestClassifier
#le = preprocessing.LabelEncoder()
#le.fit(y_train)
#y_train_le = le.transform(y_train)
#y_test_le = le.transform(y_test)

#forest = RandomForestClassifier()

#forest.fit(x_train,y_train)
#y_pred = forest.predict(x_test)
#print(f1_score(y_test_le,y_pred,average='macro'))

features = x_train.columns.tolist()
class_names = list(set(y.tolist()))

id2class = {0:'MVC', 1:'MVP', 2: 'MVVM', 3: 'NONE'}


explainer = shap.Explainer(model)
shap_values = explainer.shap_values(x_train)
def get_shap(shap_values,features=features):
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


get_shap(shap_values,features)
#shap.summary_plot(shap_values,feature_names=features,max_display=30,class_names=class_names,plot_type='bar')
