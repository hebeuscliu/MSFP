import os
import sys
import yaml
import csv
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SequentialSampler
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i >= 0:
                smiles = row['smiles']
                label = row[target]
                # label = 1
                # print(row)
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    else:
                        ValueError('task must be either regression or classification')
    print("len(smiles) -> ", len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        self.conversion = 1

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)
        else:
            print("Error -> self.task: ", self.task)
            exit()
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):

    def __init__(self,
                 batch_size, num_workers, valid_size, test_size,
                 data_path, target, task, splitting
                 ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        assert splitting in ['random']

    def get_data_loaders(self):
        dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        pred_loader = self.get_train_validation_data_loaders(dataset)

        return pred_loader

    def get_train_validation_data_loaders(self, dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(dataset)
            indices = list(range(num_train))
            # np.random.shuffle(indices)
            pred_idx = indices
            print("pred_idx -> ", len(pred_idx))
        else:
            print("Error -> wrong splitting method")
            exit()
        # define samplers for obtaining training and validation batches
        pred_sampler = SequentialSampler(pred_idx)
        pred_loader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=pred_sampler,
            num_workers=self.num_workers, drop_last=False)

        return pred_loader


class Predict(object):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.device = self._get_device()
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            print("Error -> wrong dataset task")
            exit()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]
        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        else:
            print("Error -> dataset task -> loss")
            exit()
        return loss

    def load_model(self):
        print("================= load model =======================")
        ## load model structure
        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'],
                          **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'],
                        **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        else:
            print("Error -> unknown model type")
            exit()

        return model

    def _load_pre_trained_weights(self, model):
        ## load pre-trained weights
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config[
                'fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder,
                                                 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def pred(self, model):
        print("================= prediction =======================")

        pred_loader = self.dataset.get_data_loaders()

        ## load finetuned weights
        # model_path = os.path.join('finetune','finetune',  'checkpoints', 'model.pth')
        model_path = self.config["finetuned_weights"]
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded finetuned weights with success.")

        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(pred_loader):
                data = data.to(self.device)
                __, pred = model(data)
                loss = self._step(model, data, bn)
                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)
                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)
                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())
            test_loss /= num_data

        if self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)

            guess = [1 if p > 0.25 else 0 for p in predictions[:, 1]]
            sorted_index = sorted(range(len(predictions[:, 1])),
                                  key=lambda i: predictions[:, 1][i], reverse=True)

            """
            for i in range(len(predictions[:, 1])):
                print("{:d}  {:d}  {:d} <- {:.4f}".format(
                    i, labels[i], guess[i], predictions[:,1][i]))
            """

            for i in sorted_index[:50]:
                print("{:d}  {:d}  {:d} <- {:.4f}".format(
                    i, labels[i], guess[i], predictions[:, 1][i]))

            """
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            roc_auc_guess = roc_auc_score(labels, guess)
            print('Test loss: {:.4f} Test ROC AUC: {:.4f} Guess AUC: {:.4f}'.format(
                test_loss, self.roc_auc, roc_auc_guess))
            """

            print("================ Done ===================")

            return predictions[:, 1], sorted_index

        else:
            return [], []


def main():
    res_file = '../data_and_scripts/predict-9.csv'  # 修改为当前目录下的文件名
    config = yaml.load(open("config_predict.yaml", 'r'), Loader=yaml.FullLoader)
    print(config)
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    predict = Predict(dataset, config)
    model = predict.load_model()
    preds, sorted_index = predict.pred(model)

    with open(config["dataset"]["data_path"], 'r') as fo:
        lines = fo.readlines()
    name, smiles = [], []
    for line in lines[1:]:
        items = line.strip().split(",")
        name.append(items[2])
        smiles.append(items[3])

    if len(preds) != len(name) != len(smiles):
        print("Error -> wrong length")
        exit()

    with open(res_file, 'w') as fo:
        fo.write("idx,pred,name,smiles\n")
        for index in sorted_index:
            fo.write("{},{:.4f},{},{}\n".format(index, preds[index], name[index], smiles[index]))

    print("Good Day !")
"""
    ## for BBBP
    with open(config["dataset"]["data_path"], 'r') as fo:
        lines = fo.readlines()
    if len(preds) != len(lines[1:]):
        print("Error in write results"); exit()
    with open(res_file, 'w') as fo:
        fo.write("{},{}\n".format(lines[0].strip(), "bbbp"))
        for ind, line in enumerate(lines[1:]):
            fo.write("{},{:.4f}\n".format(line.strip(), preds[ind]))
"""



if __name__ == "__main__":
    main()
