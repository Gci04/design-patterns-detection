# design-patterns-detection

## Running the project:

1) Clone Project: 
```
git clone https://github.com/Gci04/design-patterns-detection.git
```

2) Create virtualenv and install dependencies using:
```
cd design-patterns-detection && pip install -r requirements.txt
```

3) Download android apps and extract them using jupyter notebook code: `notebooks/download_prep_repos_data.ipynb`

3) Once the files are downloaded run following to train the models: 
```
python trainer.py
```

## Pipeline Configuration:

* The pipeline can be configured by following parameters in `trainer.py` file:
```
exp_setting = {
    "pooling_strategy": "sum",   # choose either (sum | max)
    "drop_multicoll": False,  # drop linearly correlated features from ck_metrics only
    "scale_before_dim_reduction": False,    # scale embeddings before applying dimensionality reduction
    "dimensionality_reduction_algo": "isomap",   # choose dimensionality reduction methodology to use: [pca, mds, isomap, tsne]
    "output_emb_dimensions" : 5,   # no of output dimensions of embeddings after apply dim reduction
    "scale_after_dim_reduction": True,   # scaling combined features (embeddings_dim_reduced + ck_metrics) before training classifiers
}
```
