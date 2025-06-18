from typing import Tuple, Dict
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
import h5py
import pickle
import json


def load_embeddings(file_path, unique_keys=None):
    with h5py.File(file_path, "r") as f:
        data = f["data"][()]
        keys = [k.decode("utf-8") for k in f["keys"][()]]
        data_dict_loaded = dict(zip(keys, data))
    if unique_keys is not None:
        new_dict = {key: data_dict_loaded[key] for key in unique_keys}
        return new_dict
    return data_dict_loaded


def get_train_test_datasets(fold, setting, dataset, target, mol_model, prot_model):
    dataset_path = f"/home/julian/DTIAM/data/{target}/" + dataset
    folds_path = dataset_path + f"/data_folds/{setting}/"
    # prot_feat = pickle.load(open(dataset_path + "/features/protein_features.pkl", "rb"))
    # train = pd.read_csv(folds_path + "train_fold_" + str(fold) + ".csv")
    # test = pd.read_csv(folds_path + "test_fold_" + str(fold) + ".csv")
    train = pd.read_csv(folds_path + "train_fold_" + str(fold) + ".csv").rename(columns={'DrugID': 'drug_id', 'TargetID': 'protein_id', 'label': 'affinity'})
    test = pd.read_csv(folds_path + "test_fold_" + str(fold) + ".csv").rename(columns={'DrugID': 'drug_id', 'TargetID': 'protein_id', 'label': 'affinity'})

    if target == 'dta':
        ligands = json.load(open(dataset_path + "/ligands_can.txt"))
        proteins = json.load(open(dataset_path + "/proteins.txt"))
    elif target == 'moa':
        ligands = pd.read_csv(f'{dataset_path}/drug_smi.csv', sep='\t').set_index('DrugID').to_dict()['smi']
        proteins = pd.read_csv(f'{dataset_path}/tar_seq.csv', sep='\t').set_index('TargetID').to_dict()['seq']
    elif target == 'dti':
        ligands = pd.read_csv(f'{dataset_path}/drug_smiles.csv', sep='\t').set_index('drug_id').to_dict()['smiles']
        proteins = pd.read_csv(f'{dataset_path}/protein_seq.csv', sep='\t').set_index('pro_id').to_dict()['seq']

    # ligands = json.load(open(dataset_path + "/ligands_can.txt"))
    # proteins = json.load(open(dataset_path + "/proteins.txt"))

    # For drugs
    train['smiles'] = train['drug_id'].apply(lambda x: ligands[x])
    test['smiles'] = test['drug_id'].apply(lambda x: ligands[x])
    mol_embs = load_embeddings(f'/home/julian/mole_embed/notebooks/DTI_benchmark/prot_mols_embeddings/{mol_model}.h5')
    train['drub_emb'] = train['smiles'].apply(lambda x: mol_embs[x])
    test['drub_emb'] = test['smiles'].apply(lambda x: mol_embs[x])
    train.drop(columns=['drug_id', 'smiles'], inplace=True)
    test.drop(columns=['drug_id', 'smiles'], inplace=True)
    del mol_embs
    # For proteins
    train['seq'] = train['protein_id'].apply(lambda x: proteins[x])
    test['seq'] = test['protein_id'].apply(lambda x: proteins[x])
    prot_embs = load_embeddings(f'/home/julian/mole_embed/notebooks/DTI_benchmark/prot_mols_embeddings/{prot_model}.h5')
    train['prot_emb'] = train['seq'].apply(lambda x: prot_embs[x])
    test['prot_emb'] = test['seq'].apply(lambda x: prot_embs[x])
    train.drop(columns=['protein_id', 'seq'], inplace=True)
    test.drop(columns=['protein_id', 'seq'], inplace=True)
    train_df = train.apply(lambda x: pd.Series(np.hstack([x['drub_emb'], x['prot_emb']])), axis=1)
    train_df['y'] = train['affinity']
    test_df = test.apply(lambda x: pd.Series(np.hstack([x['drub_emb'], x['prot_emb']])), axis=1)
    test_df['y'] = test['affinity']
    return train_df, test_df


def get_train_test_datasets_fix_prot(fold, setting, dataset='kiba', mol_model='MolE_GuacaMol_27113.ckpt', target='dta'):
    dataset_path = f"/home/julian/DTIAM/data/{target}/" + dataset
    folds_path = dataset_path + f"/data_folds/{setting}/"
    prot_feat = pickle.load(open(dataset_path + "/features/protein_features.pkl", "rb"))
    train = pd.read_csv(folds_path + "train_fold_" + str(fold) + ".csv").rename(columns={'DrugID': 'drug_id', 'TargetID': 'protein_id', 'label': 'affinity'})
    test = pd.read_csv(folds_path + "test_fold_" + str(fold) + ".csv").rename(columns={'DrugID': 'drug_id', 'TargetID': 'protein_id', 'label': 'affinity'})
    print(train.columns)
    if target == 'dta':
        ligands = json.load(open(dataset_path + "/ligands_can.txt"))
        # proteins = json.load(open(dataset_path + "/proteins.txt"))
    elif target == 'moa':
        ligands = pd.read_csv(f'{dataset_path}/drug_smi.csv', sep='\t').set_index('DrugID').to_dict()['smi']
    elif target == 'dti':
        ligands = pd.read_csv(f'{dataset_path}/drug_smiles.csv', sep='\t').set_index('drug_id').to_dict()['smiles']
    # For drugs
    train['smiles'] = train['drug_id'].apply(lambda x: ligands[x])
    test['smiles'] = test['drug_id'].apply(lambda x: ligands[x])
    MolE_embs = load_embeddings(f'/home/julian/mole_embed/notebooks/DTI_benchmark/prot_mols_embeddings/{mol_model}.h5')
    train['drub_emb'] = train['smiles'].apply(lambda x: MolE_embs[x])
    test['drub_emb'] = test['smiles'].apply(lambda x: MolE_embs[x])
    train.drop(columns=['drug_id', 'smiles'], inplace=True)
    test.drop(columns=['drug_id', 'smiles'], inplace=True)
    # For proteins
    train['prot_emb'] = train['protein_id'].apply(lambda x: prot_feat[x])
    test['prot_emb'] = test['protein_id'].apply(lambda x: prot_feat[x])
    train.drop(columns=['protein_id'], inplace=True)
    test.drop(columns=['protein_id'], inplace=True)
    train_df = train.apply(lambda x: pd.Series(np.hstack([x['drub_emb'], x['prot_emb']])), axis=1)
    train_df['y'] = train['affinity']
    test_df = test.apply(lambda x: pd.Series(np.hstack([x['drub_emb'], x['prot_emb']])), axis=1)
    test_df['y'] = test['affinity']
    return train_df, test_df


def load_data(data_path: str, fold_idx: int, comp_feat: Dict, prot_feat: Dict) -> Tuple:
    """Load training and testing data."""
    print("Loading data ...")
    train = pd.read_csv(data_path + "train_fold_" + str(fold_idx) + ".csv")
    test = pd.read_csv(data_path + "test_fold_" + str(fold_idx) + ".csv")
    train.columns = ["cid", "pid", "label"]
    test.columns = ["cid", "pid", "label"]
    return pack(train, comp_feat, prot_feat), pack(test, comp_feat, prot_feat)


def pack(data: pd.DataFrame, comp_feat: Dict, prot_feat: Dict) -> pd.DataFrame:
    """Pack compound and protein features into a dataframe."""
    vecs = []
    for i in range(len(data)):
        cid, pid = data.iloc[i, :2]
        vecs.append(list(comp_feat[str(cid)]) + list(prot_feat[pid]))
    vecs_df = pd.DataFrame(vecs)
    vecs_df["y"] = data["label"]
    return vecs_df


def roc_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the ROC AUC score."""
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def pr_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the Precision-Recall AUC score."""
    precision, recall, _ = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Root Mean Squared Error."""
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Mean Squared Error."""
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient."""
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Spearman correlation coefficient."""
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Concordance Index."""
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci
