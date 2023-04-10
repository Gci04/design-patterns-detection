from data_wrangler import find_src_path, get_java_paths
from model import embed, read_file
from tqdm import tqdm
from utils import write_pickle, read_pickle

TEST_EMBEDDINGS = 'embeddings/test_embeddings.pkl'
import os
import numpy as np


def embed_reduced_files(leave_one_out_files):
    
    # codes_all = []
    embeddings_all = []
    project_embeddings = {} 

    for file_path in leave_one_out_files:
        #         print("file_path: ", file_path)
        source_code_list = read_file(file_path)    
        source_code = ''.join(source_code_list)
        code_emb = embed(source_code)
        # embeddings_all.append(code_emb.detach().cpu().numpy())
        project_embeddings[file_path] = code_emb.detach().cpu().numpy()
        # codes_all.append(source_code)
    return project_embeddings



def make_test_embeddings(base, test_paths):
    """make one-time embeddings for test projects"""
    df = get_java_paths(base)
    df = df[df['projects'].isin(test_paths)]

    df_group = df.groupby('projects')
    
    proj_wise_embeddings = {}

    for project_name, group in tqdm(df_group, total=len(df_group)):

        # assert len(set(group['java_paths'].values.tolist())) == len(group['java_paths'].values.tolist())
        emb = embed_reduced_files(group['java_paths'].values.tolist())
        proj_wise_embeddings[project_name] = emb

    write_pickle(TEST_EMBEDDINGS, proj_wise_embeddings)


def leave_one_out(base, test_paths, y_test, clf, self):
    
    if not os.path.exists(TEST_EMBEDDINGS):
        print("Serializing embeddings...")
        make_test_embeddings(base, test_paths)

    proj_wise_embeddings = read_pickle(TEST_EMBEDDINGS)

    df = get_java_paths(base)
    df = df[df['projects'].isin(test_paths)]
    
    df_group = df.groupby('projects')
    
    proj_emb = {}
 
    for project_name, group in tqdm(df_group, total=len(df_group)):

        # assert len(set(group['java_paths'].values.tolist())) == len(group['java_paths'].values.tolist())
        
        curr_proj_emb = proj_wise_embeddings[project_name]
        single_project_files = group['java_paths'].values.tolist()
        
        sum_emb = np.sum([curr_proj_emb[file][0] for file in single_project_files], axis=0)
        proj_emb[project_name] = sum_emb


        # data = single_emb_function(curr_proj_emb, single_project_files)
    
    import pandas as pd
    
    # breakpoint()
    X_emb = self.dim_reducer.transform(np.array(list(proj_emb.values())))

    df_emb = pd.DataFrame(X_emb)
    df_emb["folder_name"] = proj_emb.keys() # project_dirs
    # breakpoint()

    df_final = self.df_ck_metrics.set_index('folder_name').join(df_emb.set_index('folder_name'), on='folder_name', how='inner').reset_index()
    
    df_final['label'] = df_final['label'].apply(lambda l: self.lbl2idx[l])
    df_final = df_final.drop(["project_name", "folder_name"], axis=1)
        
    y = df_final['label']        
    df_final.drop('label', axis=1, inplace=True)
    X = df_final.values

    # breakpoint()

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import precision_recall_fscore_support
    
    # X_test = np.array(proj_emb.values())
    y_preds = clf.predict(X)
    _, _, f, s_ = precision_recall_fscore_support(y_test, y_preds)

    print("f.tolist():: ")
    print(f.tolist())
    breakpoint()

def single_emb_function(curr_proj_emb, single_project_files):
    for i, _ in enumerate(single_project_files):    
        print(f"leaving {single_project_files[i]} file on index: {i}")
            # Create a list that excludes the i-th file
        leave_one_out_files = single_project_files[:i] + single_project_files[i+1:]
        sum_emb = np.sum([curr_proj_emb[file][0] for file in leave_one_out_files], axis=0)
        yield sum_emb