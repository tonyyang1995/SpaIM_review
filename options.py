import argparse
import os
# import torch
import scanpy as sc

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--root', type=str, default='dataset/')
        self.parser.add_argument('--dataset_name', type=str, default='nano9_1')
        self.parser.add_argument('--kfold', type=int, default=1)
        self.parser.add_argument('--batch_size', type=int, default=500)
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--val_only', type=int, default=0)
        self.parser.add_argument('--style_dim', type=int, default=1)
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--beta1', type=float, default=0.9)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--save_path', type=str, default='./SpaIM_results/')
        self.parser.add_argument('--debug_path', type=str, default='./debug/')
        self.parser.add_argument('--load_path', type=str, default='best_pcc.pth')
        self.parser.add_argument('--seed', type=int, default=1234)
        self.parser.add_argument('--parallel', type=int, default=0)
        self.parser.add_argument('--cluster', type=str, default='annotate')
        self.parser.add_argument('--gpu', type=int, default=0)
        # self.parser.add_argument('--model_layers', type=str, default='512,256')
        self.parser.add_argument('--model_layers', type=str, default='256, 512')


    def parse(self):
        self.opt = self.parser.parse_args()
        # self.opt.seq_data = os.path.join(self.opt.root, self.opt.dataset_name, 'scRNA_count.h5ad')               # for nano  change name
        # self.opt.seq_data = os.path.join(self.opt.root, self.opt.dataset_name, 'scRNA_count_leiden_0.1.h5ad')
        self.opt.seq_data = os.path.join(self.opt.root, self.opt.dataset_name, 'scRNA_count_cluster.h5ad')     # for benchmark  
        # dataset/nano5-1/scRNA_count_cluster.h5ad
        self.opt.spa_data = os.path.join(self.opt.root, self.opt.dataset_name, 'Insitu_count.h5ad')

        adata = sc.read(self.opt.seq_data)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=3)
        # self.opt.sc_dim = len(set(adata.obs['leiden']))
        # self.opt.sc_dim = len(set(adata.obs['merge_cell_type']))

        adata = sc.read(self.opt.spa_data)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=3)
        self.opt.st_dim = adata.shape[0]

        self.opt.save_path = os.path.join(self.opt.save_path, self.opt.dataset_name)
        if not os.path.exists(self.opt.save_path):
            os.makedirs(self.opt.save_path)
        
        self.opt.debug_path = os.path.join(self.opt.debug_path, self.opt.dataset_name)
        if not os.path.exists(self.opt.debug_path):
            os.makedirs(self.opt.debug_path)

        model_layers = self.opt.model_layers
        model_layers = model_layers.split(',')
        model_layers = [int(x) for x in model_layers]
        self.opt.model_layers = model_layers

        return self.opt
