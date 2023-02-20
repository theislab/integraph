import numpy as np
import scanpy as sc
import multigrate as mtg
import anndata as ad
import scipy
from scipy import sparse
import os
import matplotlib.pyplot as plt

def load_adata():
    all_samples_path = "../pp_harm_data/all_samples/"
    print("loading data..")
    rna = ad.read_h5ad(all_samples_path+"rna-pp-harm-sub.h5ad")
    adt = ad.read_h5ad(all_samples_path+"adt-pp-harm-sub.h5ad")
    cytof = ad.read_h5ad(all_samples_path+"cytof-pp-harm-sub.h5ad")
    print("rna.shape: {}, adt.shape: {}, cytof.shape: {}".format(rna.shape, adt.shape, cytof.shape))

    return rna, adt, cytof

def split_adata(rna, adt, cytof): ### split adata based on common and unique features
    intersection = adt.var.index.intersection(cytof.var.index).tolist()
    print("adt and cytof have {} common features".format(len(intersection)))
    
    adt_unique = adt[:, adt.var_names.drop(intersection)].copy()
    cytof_unique = cytof[:, cytof.var_names.drop(intersection)].copy()
    
    adt_common = adt[:, intersection].copy()
    cytof_common = cytof[:, intersection].copy()
    
    return rna, adt_unique, cytof_unique, adt_common, cytof_common

def concatenate_adata(rna, adt_unique, cytof_unique, adt_common, cytof_common):
    print("concatenating data..")
    
    combined = mtg.data.organize_multiome_anndatas(
        adatas = [[rna, None], [adt_common, cytof_common], [adt_unique, None], [None, cytof_unique]],
        layers = [[None, None], [None, None], [None, None], [None, None]],
    )
    
    return combined

def setup_combined_adata(combined):
    print("setting up the combined adata..")
    mtg.model.MultiVAE.setup_anndata(combined, categorical_covariate_keys = ['Domain'])
       
def setup_multivae(combined, l_coef,mmd):
    print("setting up the model..")
    model = mtg.model.MultiVAE(
        combined, 
        integrate_on='Domain',
        loss_coefs={'integ':l_coef},
        losses=['mse', 'mse', 'mse', 'mse'],
        mmd=mmd)
        
    return model

def model_train(model, lr):
    print("training the model..")
    model.train(lr=lr, use_gpu=True)
    
def plot_losses(model, results_path):
    model.plot_losses(results_path + "losses.jpg")
    
def save_model(model, results_path):
    print("saving the model..")
    model.save(results_path + "multigrate.dill", prefix=None, overwrite=True, save_anndata=False)
    
def get_latent_representation(model):
    print("getting latent representation for the combined adata..")
    model.get_latent_representation()
    
def write_combined(combined, results_path):
    print("writing the combined adata\n\n")
    del combined.uns['modality_lengths'] #avoids error message when writing adata containing modality lenghts
    combined.write(results_path + "combined.h5ad", compression="gzip")
    print("writing complete\n\n")
       
def load_combined(results_path):
    print("loading combined adata..\n\n")
    combined = ad.read_h5ad(results_path + "combined.h5ad")
    return combined

def load_combined(results_path):
    print("loading combined adata..\n\n")
    combined = ad.read_h5ad(results_path + "combined.h5ad")
    return combined

def compute_umap(combined, results_path):
    print("computing neighbours..\n\n")
    sc.pp.neighbors(combined, use_rep="latent", metric="cosine")
    print("computing umap..\n\n")
    sc.tl.umap(combined)
    print("writing combined adata with umap in results directory..\n\n")
    combined.write(results_path +"combined.h5ad", compression="gzip")
    print("plotting umaps..\n\n")
    os.makedirs(results_path + "umaps/", exist_ok=True)
    
    ax = sc.pl.umap(combined, color=["Annotation_major_subset", "Annotation_cell_type"], wspace=0.65, return_fig=True)
    plt.show()
    fig = ax.get_figure()
    fig.savefig(results_path+'umaps/cell_type.png')
    plt.close()
    
    ax = sc.pl.umap(combined, color=["Domain_major", "Domain"], wspace=0.65, return_fig=True)
    plt.show()
    fig = ax.get_figure()
    fig.savefig(results_path+'umaps/domain.png')
    plt.close()
    
    return combined

def main(loss_coefs=[0, 1e1, 1e2, 1e3, 1e4, 1e5],
         lr=0.00005,
         mmd='latent'):
    
    print("analysis started..\n\n")
    rna, adt, cytof = load_adata()
    (rna, adt_unique, cytof_unique, adt_common, cytof_common) = split_adata(rna, adt, cytof)
    
    for l_coef in loss_coefs:
        results_path = '../results/multigrate/trimodal/all_samples/coef_' + str(l_coef) + '/'
        os.makedirs(results_path, exist_ok=True)
        combined = concatenate_adata(rna, adt_unique, cytof_unique, adt_common, cytof_common)
        setup_combined_adata(combined)
        model = setup_multivae(combined, l_coef, mmd=mmd)
        model.to_device('cuda:0')
        model_train(model, lr=lr)
        save_model(model, results_path)
        plot_losses(model, results_path)
        get_latent_representation(model)
        write_combined(combined, results_path)
        compute_umap(combined, results_path)
        print("analysis completed\n\n")  
        
#run trimodal integration        
main(lr = 0.000005, loss_coefs=[1e3], mmd='marginal')
 