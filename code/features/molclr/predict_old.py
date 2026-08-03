import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_pred import MolTestDatasetWrapper


class Predict(object):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time+'_'+config['task_name']+'_'+config['dataset']['target']
        log_dir = os.path.join('predict', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

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
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        print("================= train =======================")
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        self._test(model, test_loader)


    def _load_pre_trained_weights(self, model):
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

    def _test(self, model, test_loader):
        print("================= test =======================")
        # model_path = os.path.join('finetune','finetune',  'checkpoints', 'model.pth')
        model_path = os.path.join('finetune', 
                    'May05_22-37-35_Abeta_p_np', 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success. ===**===")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            guess = [ 1 if p > 0.25 else 0 for p in predictions[:,1] ]
            """
            for i in range(len(predictions[:, 1])):
                print("{:d}  {:d}  {:d} <- {:.4f}".format(
                    i, labels[i], guess[i], predictions[:,1][i]))
            """
            sorted_index = sorted(range(len(predictions[:,1])), 
                             key=lambda i:predictions[:,1][i], reverse=True)
            for i in sorted_index:
                print("{:d}  {:d}  {:d} <- {:.4f}".format(
                    i, labels[i], guess[i], predictions[:,1][i]))
            """
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            roc_auc_guess = roc_auc_score(labels, guess)
            print('Test loss: {:.4f} Test ROC AUC: {:.4f} Guess AUC: {:.4f}'.format(
                test_loss, self.roc_auc, roc_auc_guess))
            """
            with open("data/Abeta/results.csv", 'w') as fo:
                fo.write("index,pred\n")
                for i in range(len(predictions[:, 1])):
                    fo.write("{:d},{:.4f}\n".format(i, predictions[:,1][i]))

            print("================ Done ===================")



def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    predict = Predict(dataset, config)
    predict.train()


if __name__ == "__main__":
    config = yaml.load(open("config_predict.yaml", 'r'), Loader=yaml.FullLoader)
    print(config)
    main(config)

