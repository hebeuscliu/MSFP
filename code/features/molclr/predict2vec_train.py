
import os
import sys
import yaml
import numpy as np
import torch
from torch import nn
from torch_geometric.data import DataLoader
import pandas as pd
import csv  # 新增：用于读取原始 CSV

# 假设这个路径是正确的，包含了 MolTestDatasetWrapper
from dataset.dataset_pred import MolTestDatasetWrapper

# --- 【配置常量】 ---
DEFAULT_CONFIG = {
    'batch_size': 32,
    'model_type': 'gcn',
    'fine_tune_from': 'pretrained_gcn',
    'gpu': 'cuda:0',
    'task_name': 'BC',
    'model': {
        'num_layer': 5,
        'emb_dim': 300,
        'feat_dim': 512,
        'drop_ratio': 0.3,
        'pool': 'mean'
    },
    'finetuned_weights': './finetune/Jul05_00-47-33_BC_p_n_9/checkpoints/model.pth',
    'dataset': {
        'num_workers': 4,
        'valid_size': 0.1,
        'test_size': 0.1,
        'splitting': 'random',
        'task': 'classification',
        'data_path': './data/BC/data4finetune_9.csv',
        'target': 'p_n'
    }
}
OUTPUT_VEC_FILE = 'extracted_molclr_vectors_data_9.npy'
OUTPUT_SMILES_FILE = 'extracted_molclr_smiles_data_9.csv'


# --------------------

# Predict 类保持不变，因为它成功完成了提取任务

# ... (Predict 类的定义，与上次提供的代码完全一致) ...
class Predict(object):
    # ... (省略内部方法，保持不变) ...
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.device = self._get_device()

    def _get_device(self):
        gpu_config = self.config.get('gpu', 'cpu')
        if torch.cuda.is_available() and gpu_config != 'cpu':
            device = gpu_config
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def _load_pre_trained_weights(self, model):
        try:
            fine_tune_from = self.config.get('fine_tune_from', 'pretrained_gcn')
            checkpoints_folder = os.path.join('./ckpt', fine_tune_from, 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded pre-trained GNN Encoder weights with success.")
        except Exception as e:
            print(f"Warning: Failed to load pre-trained weights ({e}).")
        return model

    def load_model(self):
        print("================= Load Model =======================")
        model_params = self.config["model"]
        task = self.config['dataset']['task']

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune import GINet
            ModelClass = GINet
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            ModelClass = GCN
        else:
            print(f"Error: Unknown model type {self.config['model_type']}")
            sys.exit(1)

        model = ModelClass(task=task, **model_params).to(self.device)
        model = self._load_pre_trained_weights(model)

        model_path = self.config.get("finetuned_weights", DEFAULT_CONFIG['finetuned_weights'])

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded fine-tuned GNN Encoder weights from {model_path} successfully.")
        except FileNotFoundError:
            print(f"Error: Fine-tuned weights not found at {model_path}. Extraction aborted.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading fine-tuned weights: {e}. Check if architecture matches.")
            sys.exit(1)

        model.eval()
        return model

    def extract_vec(self, model):
        """执行向量提取的核心逻辑，并返回提取后的向量和相应的 SMILES 列表。"""
        print("================= Extracting Vectors =======================")

        loaders = self.dataset.get_data_loaders()

        if loaders is None or len(loaders) != 3:
            print("Error: MolTestDatasetWrapper did not return the expected three loaders.")
            return None, None

        pred_loader = loaders[2]

        if pred_loader is None:
            print("Error: Prediction DataLoader is None.")
            return None, None

        # ----------------------------------------------------
        # 核心：获取 MolTestDataset 实例以提取 SMILES
        test_dataset = pred_loader.dataset

        try:
            # 这一步依赖于您的 MolTestDataset 类有一个 self.smiles_data 属性
            extracted_smiles = test_dataset.smiles_data
        except AttributeError:
            print("Warning: Could not access 'smiles_data' attribute from the dataset. Cannot save SMILES list.")
            extracted_smiles = None
        # ----------------------------------------------------

        all_embeddings = []
        with torch.no_grad():
            for data in pred_loader:
                data = data.to(self.device)
                embedding, _ = model(data)
                all_embeddings.append(embedding.cpu().numpy())

        if not all_embeddings:
            print("Warning: No embeddings were extracted.")
            return None, None

        final_vectors = np.vstack(all_embeddings)
        print(
            f"Extraction complete. Total vectors: {final_vectors.shape[0]}. Vector dimension: {final_vectors.shape[1]}")

        np.save(OUTPUT_VEC_FILE, final_vectors)
        print(f"Vectors saved to {OUTPUT_VEC_FILE}")

        return final_vectors, extracted_smiles


def get_original_smiles(data_path):
    """读取原始 CSV 文件中的所有 SMILES，不进行过滤。"""
    original_smiles = []
    try:
        with open(data_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # 假设 SMILES 列名是 'smiles'
                original_smiles.append(row['smiles'])
    except Exception as e:
        print(f"Error reading original CSV file at {data_path}: {e}")
        return []
    return original_smiles


def main_extract_vec(config):
    data_path = config['dataset'].get('data_path', DEFAULT_CONFIG['dataset']['data_path'])
    config['dataset']['data_path'] = data_path

    # --- 【新增对比逻辑】 ---
    original_smiles = get_original_smiles(data_path)

    # 1. 初始化数据包装器
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

    # 2. 初始化 Predict 类
    predictor = Predict(dataset, config)

    # 3. 加载模型
    model = predictor.load_model()

    # 4. 提取向量和 MolCLR 实际处理的 SMILES 列表
    final_vectors, extracted_smiles = predictor.extract_vec(model)

    # 5. 对比和输出缺失的 SMILES
    if extracted_smiles and len(original_smiles) != len(extracted_smiles):
        original_set = set(original_smiles)
        extracted_set = set(extracted_smiles)

        missing_smiles = list(original_set - extracted_set)

        if missing_smiles:
            print("\n=======================================================")
            print(f"🚨 发现缺失 SMILES: 总数 {len(original_smiles)}, 提取数 {len(extracted_smiles)} 🚨")
            print(f"被 MolCLR 过滤掉的 SMILES 是 (共 {len(missing_smiles)} 条):")
            for smiles in missing_smiles:
                # 尝试找到缺失 SMILES 的原始行号 (可能较慢)
                try:
                    index = original_smiles.index(smiles) + 1  # +1 for 1-based index
                    print(f" - [Row {index}] {smiles}")
                except ValueError:
                    print(f" - {smiles} (Row Index Unknown)")
            print("=======================================================\n")

        # 6. 保存提取的 SMILES 列表
        df_smiles = pd.DataFrame({'smiles': extracted_smiles})
        df_smiles.to_csv(OUTPUT_SMILES_FILE, index=False)
        print(f"Corresponding SMILES list saved to {OUTPUT_SMILES_FILE}")
    elif extracted_smiles:
        # 如果数量一致，也保存列表
        df_smiles = pd.DataFrame({'smiles': extracted_smiles})
        df_smiles.to_csv(OUTPUT_SMILES_FILE, index=False)
        print(f"Corresponding SMILES list saved to {OUTPUT_SMILES_FILE}")
    else:
        print("SMILES list not available to save.")


if __name__ == "__main__":
    try:
        config = yaml.load(open("config_predict.yaml", 'r'), Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("Warning: config_predict.yaml not found. Using default internal configuration.")
        config = DEFAULT_CONFIG

    print(config)
    main_extract_vec(config)