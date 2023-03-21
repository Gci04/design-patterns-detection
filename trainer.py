"""Driver code: Before running this one needs to download and preprocess coach_source code and
 that can be accomplished by running `download_prep_repos_data.ipynb` code"""
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from data_wrangler import find_src_path, get_java_paths
from utils import scale_data
from termcolor import colored

from utils import mds, isomap, pca


def drop_multicollinear(df_ck_metrics, COLLINEARITY_THRESHOLD = 0.98):
    """Drop everything more COLLINEARITY_THRESHOLD collinear"""
    corr_matrix = df_ck_metrics.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= COLLINEARITY_THRESHOLD)]
    print("Dropping features: ", len(to_drop))
    # Dropping highly correlated columns
    df_ck_metrics = df_ck_metrics.loc[:, ~df_ck_metrics.columns.isin(to_drop)]
    return df_ck_metrics


def read_embeddings(base = 'embeddings', pooling = 'sum'):
    """
    Read serialized embeddings fro disk
    If you want to regenerate these embeddings use `EmbeddingsGeneration` class from embeddings_gen.py
    """
    import shelve
    assert pooling in ('sum', 'max'), "Pooling must be one of the following: ['sum', 'max']"
    KEYSPACE = f"proj_embeddings_filtered_{pooling}pool"
    source_path = f"{base}/features_dict_{pooling}pool"
    with shelve.open(source_path) as db:
        proj_embeddings_filtered, lbl2idx, y = db[KEYSPACE]
    return proj_embeddings_filtered, lbl2idx, y


def log_results(name, X_shape, m_scores, std_scores):
    with open('log.dat', "a") as file:
        # should have atleast 0.
        if (m_scores > 0.6).sum() > 1 and (m_scores > 0.5).sum() > 2:
            breakpoint()
            file.write(f"X.shape: {X_shape}\n")
            file.write(f"results of: {name}: \n" )
            file.write("Mean: [" + ", ".join(str(s) for s in m_scores) + "]\n")
            file.write("Std_dev: [" + ", ".join(str(s) for s in std_scores) + "]")
            file.write("\n\n")


BASE = Path('./coach_repos_zip')

assert "./coach_repos_zip/2048-android-master/src" == find_src_path('./coach_repos_zip/2048-android-master')

def read_projects():
    """read project codes to keep track of all the .java files against each project"""
    df = get_java_paths(BASE)
    df['projects'] = df['projects'].str.lower().str.strip().values

    print("df.groupby('projects').count()::")
    print(df.groupby('projects').count())
    return df


# manipulate 
exp_setting = {
    "drop_multicoll": False,  # drop multi-collinearity
    "scale_before_dim_reduction": False, 
    "dimensionality_reduction_algo": "mds",   # choose dimensionality reduction methodology to use: [pca, mds, isomap]
    "output_emb_dimensions" : 30,
    "scale_after_dim_reduction": False,
}

class Trainer:

    def __init__(self, ck_features_fname = "ck_features.csv") -> None:
        
        self.df_ck_metrics = pd.read_csv(ck_features_fname)

        if exp_setting['drop_multicoll']:
            self.df_ck_metrics = drop_multicollinear(self.df_ck_metrics)
        
        self.df = read_projects()
        self._preprocess()
        proj_embeddings, self.lbl2idx, target = read_embeddings()
        self.idx2lbl = dict(zip(self.lbl2idx.values(), self.lbl2idx.keys()))
        self.X_raw_emb = tf.convert_to_tensor(list(proj_embeddings.values()))
        self.X, self.y = self._apply_criterion(proj_embeddings.keys())


    def train(self):
        from algos import logistic_train, dt, lda, svm, rf, knn, light, catboost,  catboost2, statified_results
        algo_sigs = [logistic_train, dt, lda, svm, rf, knn, catboost,  catboost2, light]
        for clf_sig in algo_sigs:
            clf_name = clf_sig.__name__
            print("X.shape: ", self.X.shape)
            mean_scores, std_scores = statified_results( self.X, self.y, clf_sig)
            print(colored(f'{clf_name} results:', 'red'))
            print(mean_scores, std_scores)
            log_results(clf_name, self.X.shape, mean_scores, std_scores)

    def _apply_criterion(self, project_dirs):
        
        dim_reduction = {
            "pca": pca,
            "mds": mds,
            "isomap": isomap,
        }
        X_local = self.X_raw_emb
        if exp_setting['scale_before_dim_reduction']:
            X_local = scale_data(X_local)
        
        out_emb_dims = exp_setting['output_emb_dimensions']
        X_local = dim_reduction[exp_setting['dimensionality_reduction_algo']](X_local, out_emb_dims)
        
        df_emb = pd.DataFrame(X_local)
        df_emb["folder_name"] = project_dirs

        # Concatenating features (embeddings + ck_metrics)
        df_final = self.df_ck_metrics.set_index('folder_name').join(df_emb.set_index('folder_name'), on='folder_name').reset_index()
        df_final = df_final.drop(["project_name", "folder_name"], axis=1)
        df_final['label'] = df_final['label'].apply(lambda l: self.lbl2idx[l])

        y = df_final['label']        
        df_final.drop('label', axis=1, inplace=True)
        X = df_final.values
        
        if exp_setting['scale_after_dim_reduction']:
            X = scale_data(X)
        return X, y

    def _preprocess(self):
        self.df_ck_metrics['folder_name'] = self.df_ck_metrics['project_name'].str.lower().str.strip().values
        assert len(self.df_ck_metrics[self.df_ck_metrics['folder_name'].str.startswith('android-mvvm')]) == 2
        # proj_to_label = dict(zip(self.df_ck_metrics['folder_name'], self.df_ck_metrics['label']))
        proj_dirs = {proj:i  for i, (proj, _) in enumerate(self.df.groupby('projects')) if proj not in ('archi-master', 'archi-master-master')}
        assert len(proj_dirs) == 69


if __name__ == "__main__":
    Trainer().train()

