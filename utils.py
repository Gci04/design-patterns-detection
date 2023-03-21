from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.manifold import MDS


def scale_data(data):
    scaler = StandardScaler()
    X_scaled_ = scaler.fit_transform(data)
    return X_scaled_


def pca(X_data, out_comp):
    # kernel ={‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
    pca=KernelPCA( n_components=out_comp, kernel="poly")
    return pca.fit_transform(X_data)


def mds(X_data, out_comp):
    mds = MDS(n_components=out_comp)
    return mds.fit_transform(X_data)


def isomap(X_data, out_comp):
    iso = Isomap(n_components=out_comp)
    return iso.fit_transform(X_data)