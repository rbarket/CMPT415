from __future__ import division
from time import time
from pathlib import Path
####
import ItClust as ic
import scanpy.api as sc
import scanpy
import anndata as ann
import os
import pandas as pd
import json
import numpy as np
import warnings


def read_data(path, tissue, stats=False):
    path = Path(path)
    anndata = ann.read_csv(path / tissue)

    anndata.obs['barcode'] = anndata.obs.index

    annotations = pd.read_csv(Path(path.parent, 'annotations_facs.csv'))
    annotations.index = annotations.cell
    cell_ontology_dict = annotations['cell_ontology_class'].to_dict()
    anndata.obs['celltype'] = anndata.obs.index.map(cell_ontology_dict)
    n_cells = anndata.shape[0]
    anndata = anndata[~anndata.obs.celltype.isna()]
    n_annotated_cells = anndata.shape[0]

    anndata.var_names_make_unique(join="-")

    if stats:
        return anndata, {'n_cells': n_cells, 'n_annotated_cells': n_annotated_cells}
    else:
        return anndata


def ItClust_train(target_name, source_list):
    print(f'training target {target_name}')

    dir_path = '../../../data/FACS'

    name_dict = {
        'bladder': 'Bladder-counts.csv',
        'kidney': 'Kidney-counts.csv',
        'lung': 'Lung-counts.csv',
        'heart': 'Heart-counts.csv',
        'aorta': 'Aorta-counts.csv',
        'limb_muscle': 'Limb_Muscle-counts.csv',
        'diaphragm': 'Diaphragm-counts.csv',
        'trachea': 'Trachea-counts.csv',
        'tongue': 'Tongue-counts.csv',
        'thymus': 'Thymus-counts.csv',
        'spleen': 'Spleen-counts.csv',
        'skin': 'Skin-counts.csv',
        'pancreas': 'Pancreas-counts.csv',
        'marrow': 'Marrow-counts.csv',
        'mammary_gland': 'Mammary_Gland-counts.csv',
        'fat': 'Fat-counts.csv',
        'large_intestine': 'Large_Intestine-counts.csv',
        'brain_myeloid': 'Brain_Myeloid-counts.csv',
        'brain_non_myeloid': 'Brain_Non-Myeloid-counts.csv',
        'liver': 'Liver-counts.csv',
    }

    # merge all source data together
    for i in range(len(source_list)):
        if i == 0:
            adata = read_data(dir_path, name_dict[source_list[i]])
        else:
            adata = adata.concatenate(read_data(dir_path, name_dict[source_list[i]]))

    print('done reading')

    target = read_data(dir_path, name_dict[target_name])

    clf = ic.transfer_learning_clf()
    clf.fit(adata, target)
    print(f'done training {target_name}')

    # Write results as h5ad
    pred, prob, cell_type_pred = clf.predict(write=False)

    clf.adata_test.write('/scratch/rbarket/ItClust_Results/' + target_name + 'ItClust.h5ad')

    outfile = open('/scratch/rbarket/ItClust_Results/' + target_name + 'Final.json', 'w')  # for compute canada
    json.dump(cell_type_pred, outfile)
    outfile.close()

    del adata
    del clf


print('hello this is a test')

tissues = ['bladder',
           'kidney',
           'lung',
           'heart',
           'aorta',
           'limb_muscle',
           'diaphragm',
           'trachea',
           'tongue',
           'thymus',
           'spleen',
           'skin',
           'pancreas',
           'marrow',
           'mammary_gland',
           'fat',
           'large_intestine',
           'brain_myeloid',
           'brain_non_myeloid',
           'liver']

for target in tissues:
    source_list = [tissue for tissue in tissues if tissue != target]
    ItClust_train(target, source_list)

