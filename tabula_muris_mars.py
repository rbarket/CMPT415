import torch
import sys
import numpy as np
import pandas as pd
import scanpy.api as sc
import anndata as ann
import json

sys.path.append("../")
from pathlib import Path
from args_parser import get_parser
import os
import pickle
from model.mars import MARS
from model.experiment_dataset import ExperimentDataset
from utils import read_data
from utils import preprocess_data
import warnings


warnings.filterwarnings('ignore')
sys.path.append("../")
warnings.filterwarnings('ignore')
params, unknown = get_parser().parse_known_args()
# if torch.cuda.is_available() and not params.cuda:
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
params.device = device
# params.device = 'cuda'

dir_path = '../data/FACS'  # change this

bladder = read_data(dir_path, 'Bladder-counts.csv')
kidney = read_data(dir_path, 'Kidney-counts.csv')
lung = read_data(dir_path, 'Lung-counts.csv')
heart = read_data(dir_path, 'Heart-counts.csv')
aorta = read_data(dir_path, 'Aorta-counts.csv')
limb_muscle = read_data(dir_path, 'Limb_Muscle-counts.csv')
diaphragm = read_data(dir_path, 'Diaphragm-counts.csv')
trachea = read_data(dir_path, 'Trachea-counts.csv')
tongue = read_data(dir_path, 'Tongue-counts.csv')
thymus = read_data(dir_path, 'Thymus-counts.csv')
spleen = read_data(dir_path, 'Spleen-counts.csv')
skin = read_data(dir_path, 'Skin-counts.csv')
pancreas = read_data(dir_path, 'Pancreas-counts.csv')
marrow = read_data(dir_path, 'Marrow-counts.csv')
mammary_gland = read_data(dir_path, 'Mammary_Gland-counts.csv')
fat = read_data(dir_path, 'Fat-counts.csv')
large_intestine = read_data(dir_path, 'Large_Intestine-counts.csv')
brain_myeloid = read_data(dir_path, 'Brain_Myeloid-counts.csv')
brain_non_myeloid = read_data(dir_path, 'Brain_Non-Myeloid-counts.csv')
liver = read_data(dir_path, 'Liver-counts.csv')
print('done reading')

adata_dict = {'bladder': bladder, 'kidney': kidney, 'lung': lung, 'heart': heart,
              'aorta': aorta, 'limb_muscle': limb_muscle, 'diaphragm': diaphragm, 'trachea': trachea,
              'brain_non_myeloid': brain_non_myeloid, 'tongue': tongue, 'thymus': thymus, 'spleen': spleen,
              'pancreas': pancreas, 'marrow': marrow, 'mammary_gland': mammary_gland, 'fat': fat,
              'large_intestine': large_intestine, 'brain_myeloid': brain_non_myeloid, 'skin': skin, 'liver': liver
              }

for adata in adata_dict.values():
    preprocess_data(adata)

# created numbered index for all the labels
obs_list = [df.obs for df in adata_dict.values()]
obs_str = pd.concat(obs_list)
unique_types = set(obs_str.celltype)
celltype_dict = {}
celltype_dict_reveresed = {}
for i, celltype in enumerate(unique_types):
    celltype_dict[celltype] = i
    celltype_dict_reveresed[i] = celltype

for name, data in adata_dict.items():
    print(f'training unannotated {name}')
    n = len(set(data.obs.celltype))
    # annotated data is every tissue except current one in loop
    annotated = [ExperimentDataset(ann.X,
                                   ann.obs_names,
                                   ann.var_names,
                                   tissue,
                                   ann.obs['celltype'].replace(celltype_dict).values)
                 for tissue, ann in adata_dict.items() if tissue != name]

    # unannotated data is the current one in loop
    unannotated = ExperimentDataset(data.X,
                                    data.obs_names,
                                    data.var_names,
                                    name,
                                    data.obs['celltype'].replace(celltype_dict).values)

    mars = MARS(n, params, annotated, unannotated, hid_dim_1=1000, hid_dim_2=100)
    adata, landmarks, scores = mars.train(evaluation_mode=True, save_all_embeddings=True)  # evaluation mode
    print(f'done training {name}')
    n_landmarks = [landmark.numpy().tolist() for landmark in landmarks]

    # Predict names of clusters
    names = mars.name_cell_types(adata, landmarks, celltype_dict_reveresed)
    guess = {int(index): name[-1][0] for index, name in names.items()}  # take the best guess

    # Rank Genes
    sc.tl.rank_genes_groups(adata, 'MARS_labels', method='t-test')
    ranked_genes = adata.uns['rank_genes_groups']['names'].tolist()  # originally rec.array
    gene_scores = adata.uns['rank_genes_groups']['scores'].tolist()

    # Save results to scratch folder
    adata.write('/scratch/rbarket/mars_results/' + name + '.h5ad')
    outfile = open('/scratch/rbarket/mars_results/' + name + '.json', 'w')  # for compute canada
    adata.write('test.h5ad')
    outfile = open('test.json', 'w')
    json.dump({'landmarks': n_landmarks, 'scores': scores,
               'cluster_names': guess, 'markers': ranked_genes, 'marker_scores': gene_scores},
              outfile)
    outfile.close()

print('done training')

