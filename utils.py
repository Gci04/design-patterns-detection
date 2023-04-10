from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.manifold import MDS, TSNE
import pickle


def scale_data(data):
    scaler = StandardScaler()
    X_scaled_ = scaler.fit_transform(data)
    return X_scaled_, scaler


def pca(X_data, out_comp):
    # kernel ={‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
    pca=KernelPCA( n_components=out_comp, kernel="poly")
    return pca.fit_transform(X_data), pca


def mds(X_data, out_comp):
    mds = MDS(n_components=out_comp)
    return mds.fit_transform(X_data), mds


def isomap(X_data, out_comp):
    iso = Isomap(n_components=out_comp)
    return iso.fit_transform(X_data), iso


def tsne(X_data, out_comp):
    tsne = TSNE(n_components=out_comp)
    return tsne.fit_transform(X_data), tsne
    


def write_pickle(filename, data):
    # Open the file for writing in binary mode
    with open(filename, 'wb') as f:
        # Use the pickle module to dump the dictionary object to the file
        pickle.dump(data, f)


def read_pickle(filename):
    # Load the dictionary from the pickle file
    with open(filename, 'rb') as f:
        # Use the pickle module to load the dictionary object from the file
        loaded_data = pickle.load(f)
    return loaded_data