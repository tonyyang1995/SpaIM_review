import scanpy as sc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def adata_to_cluster_expression(adata, scale=True):
    # unique_labels = np.unique(cluster_label)
    value_counts = adata.obs['leiden'].value_counts(normalize=True)
    unique_labels = value_counts.index
    new_obs = pd.DataFrame({'leiden': unique_labels})
    print(unique_labels)
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)
    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, l, in enumerate(unique_labels):
        X_new[index] = adata[adata.obs['leiden'] == l].X.sum(axis=0)
    adata_ret.X = X_new
    return adata_ret

# root = 'dataset/benchmark_datasets/DataUpload'
# dataset_name = 'Dataset'
# for k in range(45, 46):
#     dataset = dataset_name + str(k)
#     print(dataset)
#     rna_file = os.path.join(root, dataset, 'scRNA_count.txt')
#     sc_data = sc.read(rna_file, sep='\t', first_column_names=True).T
#     sc_data.write(os.path.join(root, dataset, 'scRNA_count.h5ad'))

#     Spatial_file = os.path.join(root, dataset, 'Insitu_count.txt')
#     st_data = sc.read(Spatial_file, sep='\t')
#     st_data.write(os.path.join(root, dataset, 'Insitu_count.h5ad'))

# root = 'dataset/benchmark_datasets/DataUpload'
# dataset_name = 'Dataset'
# for k in tqdm(range(1, 46)):
#     dataset = dataset_name + str(k)

#     scadata = sc.read(os.path.join(root, dataset, 'scRNA_count.h5ad'))

#     # generate annotation
#     adata_label = scadata.copy()
#     sc.pp.normalize_total(adata_label)
#     sc.pp.log1p(adata_label)
#     sc.pp.highly_variable_genes(adata_label)
#     adata_label = adata_label[:, adata_label.var.highly_variable]
#     sc.pp.scale(adata_label, max_value=10)
#     sc.tl.pca(adata_label)
#     sc.pp.neighbors(adata_label)
#     sc.tl.leiden(adata_label, resolution=0.5, random_state=1234)
#     scadata.obs['leiden'] = adata_label.obs['leiden']

#     scadata.write(os.path.join(root, dataset, 'scRNA_count_cluster.h5ad'))

root = 'dataset/'
dataset_name = 'Dataset46'

scadata = sc.read(os.path.join(root, dataset_name, 'scRNA_count_cluster.h5ad'))
adata_label = scadata.copy()
sc.pp.normalize_total(adata_label)
sc.pp.log1p(adata_label)
sc.pp.highly_variable_genes(adata_label)
adata_label = adata_label[:, adata_label.var.highly_variable]
sc.pp.scale(adata_label, max_value=10)
sc.tl.pca(adata_label)
sc.pp.neighbors(adata_label)
for res in ['0.0005', '0.005', '0.02', '0.10', '0.30', '0.50', '0.70', '0.90']:
    print(res)
    # generate annotation
    sc.tl.leiden(adata_label, resolution=float(res), random_state=1234)
    print(len(set(adata_label.obs['leiden'])))
    scadata.obs['leiden_%s'%(res)] = adata_label.obs['leiden']

scadata.write(os.path.join(root, dataset_name, 'scRNA_count_cluster.h5ad'))