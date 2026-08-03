import os
import sys
import yaml
import numpy as np
import torch
from torch import nn
from torch_geometric.data import DataLoader
import pandas as pd
import csv
from rdkit import Chem  # 仅用于参考

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
    'finetuned_weights': './finetune/Jul03_16-24-55_BC_p_n_0/checkpoints/model.pth',
    'dataset': {
        'num_workers': 4,
        'valid_size': 0.1,
        'test_size': 0.1,
        'splitting': 'random',
        'task': 'classification',
        'data_path': './data/BC/data4finetune_0.csv',
        'target': 'p_n',
        'smiles_col': 'smiles'
    }
}
OUTPUT_VEC_FILE = 'extracted_molclr_vectors_data_0.npy'
OUTPUT_SMILES_FILE = 'molclr_aligned_data_780.csv'
SMILES_COL = DEFAULT_CONFIG['dataset']['smiles_col']


# --------------------

class Predict(object):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.device = self._get_device()
        self.smiles_col = self.config['dataset'].get('smiles_col', SMILES_COL)

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

        # 假设所有用于预测的数据都在 test_loader (索引 2)
        pred_loader = loaders[2]
        if pred_loader is None:
            print("Error: Prediction DataLoader is None.")
            return None, None

        # ----------------------------------------------------
        # 🌟 核心修正：直接从底层 Dataset 实例获取有效 SMILES 列表
        test_dataset = pred_loader.dataset
        extracted_smiles_in_order = []
        smiles_found = False

        # 尝试 MolCLR 库中常见的 SMILES 存储属性名：
        smiles_attributes = [self.smiles_col, 'smiles', 'raw_smiles', 'df', 'data', 'dataset']

        # ----------------------------------------------------
        # 🌟 临时调试代码块 🌟
        print("\n--- 临时调试：尝试定位 SMILES 列表 ---")
        for attr in smiles_attributes:
            if hasattr(test_dataset, attr):
                data = getattr(test_dataset, attr)
                count = -1

                if isinstance(data, pd.DataFrame) and self.smiles_col in data.columns:
                    count = len(data)
                    if count == 780:
                        extracted_smiles_in_order = data[self.smiles_col].tolist()
                        smiles_found = True
                        print(f"  *** 成功定位到 780 条有效 SMILES 的属性: '{attr}' (DataFrame) ***")
                        break
                elif isinstance(data, list):
                    count = len(data)
                    if count == 780 and all(isinstance(s, str) for s in data[:10]):
                        extracted_smiles_in_order = data
                        smiles_found = True
                        print(f"  *** 成功定位到 780 条有效 SMILES 的属性: '{attr}' (List) ***")
                        break
                elif hasattr(data, 'smiles') and isinstance(data.smiles, list):
                    count = len(data.smiles)
                    if count == 780:
                        extracted_smiles_in_order = data.smiles
                        smiles_found = True
                        print(f"  *** 成功定位到 780 条有效 SMILES 的属性: '{attr}' (PyG InMemory) ***")
                        break

                if count != -1:
                    print(f"  [Attr: {attr}] 找到列表/DataFrame，长度为: {count}")

        if smiles_found:
            print("--- 临时调试结束：已成功定位 ---")
        else:
            print("--- 临时调试结束：未能定位到长度为 780 的 SMILES 列表 ---")
        # ----------------------------------------------------

        if not smiles_found:
            print(f"警告：无法从 Dataset 中直接获取 SMILES 列表。将尝试从 Batch 对象中提取 (可能不完整)。")

        # 向量提取流程开始
        all_embeddings = []
        with torch.no_grad():
            for data in pred_loader:
                data = data.to(self.device)
                embedding, _ = model(data)
                all_embeddings.append(embedding.cpu().numpy())

                # 备用：如果上面失败，尝试从 Batch 中获取 (只在上面找不到时，且只能作为辅助尝试)
                if not smiles_found and hasattr(data, self.smiles_col):
                    smiles_list = getattr(data, self.smiles_col)
                    if isinstance(smiles_list, list):
                        extracted_smiles_in_order.extend(smiles_list)

        if not all_embeddings:
            print("Warning: No embeddings were extracted.")
            return None, None

        final_vectors = np.vstack(all_embeddings)
        print(
            f"Extraction complete. Total vectors: {final_vectors.shape[0]}. Vector dimension: {final_vectors.shape[1]}")

        np.save(OUTPUT_VEC_FILE, final_vectors)
        print(f"Vectors saved to {OUTPUT_VEC_FILE}")

        # 检查 SMILES 列表是否匹配
        if len(extracted_smiles_in_order) > 0 and len(final_vectors) != len(extracted_smiles_in_order):
            print(
                f"严重警告: 向量数 ({len(final_vectors)}) 与提取的 SMILES 数 ({len(extracted_smiles_in_order)}) 不匹配。")
            return final_vectors, None

        if len(extracted_smiles_in_order) == 0:
            print("警告: 无法获取精确对齐的 SMILES 列表。")
            return final_vectors, None

        return final_vectors, extracted_smiles_in_order


def get_original_smiles_and_data(data_path):
    """读取原始 CSV 文件中的所有数据，用于后续对齐。"""
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error reading original CSV file at {data_path}: {e}")
        return pd.DataFrame()


def main_extract_vec(config):
    data_path = config['dataset'].get('data_path', DEFAULT_CONFIG['dataset']['data_path'])

    # 1. 读取原始数据 (包含所有 SMILES)
    df_original = get_original_smiles_and_data(data_path)
    all_original_smiles = set(df_original[SMILES_COL].tolist())

    # 🌟 关键修正：复制配置字典并移除 MolTestDatasetWrapper 不识别的参数
    dataset_config = config['dataset'].copy()
    if 'smiles_col' in dataset_config:
        # MolTestDatasetWrapper 无法识别此参数，必须移除
        del dataset_config['smiles_col']

        # 2. 初始化数据包装器
    # 传入的参数现在是 MolTestDatasetWrapper 预期接受的参数
    dataset = MolTestDatasetWrapper(config['batch_size'], **dataset_config)

    # 3. 初始化 Predict 类 & 加载模型
    # 注意：Predict 类内部保留了 self.smiles_col 用于读取原始文件
    predictor = Predict(dataset, config)
    model = predictor.load_model()

    # 4. 提取向量和 MolCLR 实际处理的 SMILES 列表 (期望 780 条)
    final_vectors, extracted_smiles = predictor.extract_vec(model)

    if final_vectors is None:
        print("向量提取失败，中止报告。")
        return

    if extracted_smiles is None:
        print("由于 SMILES 列表获取失败，无法生成对齐文件和缺失报告。")
        return

    # 5. 对比和输出缺失的 SMILES
    print("================= Missing SMILES Report =======================")

    extracted_set = set(extracted_smiles)
    missing_smiles = list(all_original_smiles - extracted_set)

    print(f"原始有效 SMILES 总数: {len(all_original_smiles)}")
    print(f"实际提取向量数: {final_vectors.shape[0]}")

    if missing_smiles:
        print("\n---------------------------------------------------------")
        print(f"!!! 警告：共丢失 {len(missing_smiles)} 条样本 !!!")

        # 详细打印丢失的 SMILES
        for smiles in missing_smiles:
            original_index = df_original[df_original[SMILES_COL] == smiles].index.tolist()
            # 打印原始行号 (行号 = 索引 + 2，因为索引从 0 开始，且有表头)
            print(f"- SMILES: {smiles} (原始行号: {original_index[0] + 2 if original_index else 'N/A'})")

        # 保存丢失的 SMILES 到 CSV 文件
        pd.DataFrame({'missing_smiles': missing_smiles}).to_csv('missing_molclr_smiles.csv', index=False)
        print("丢失的 SMILES 已保存到 missing_molclr_smiles.csv")
        print("---------------------------------------------------------")
    else:
        print("\n所有原始 SMILES 均已成功处理。")

    # 6. 保存提取的 SMILES 及其标签（对齐 UniMol-2）
    df_extracted = pd.DataFrame({SMILES_COL: extracted_smiles})

    # 合并以获取标签和其余信息，并保持提取顺序
    df_aligned = pd.merge(df_extracted, df_original, on=SMILES_COL, how='left')

    df_aligned['molclr_vector_index'] = df_aligned.index  # 增加索引列方便后续检查

    df_aligned.to_csv(OUTPUT_SMILES_FILE, index=False)
    print(f"\n{len(df_aligned)} 条有效样本和标签已按提取顺序保存到 {OUTPUT_SMILES_FILE}")
    print("下一步：使用此文件中的 SMILES 提取 UniMol-2 向量！")


if __name__ == "__main__":
    try:
        config = yaml.load(open("config_predict.yaml", 'r'), Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("Warning: config_predict.yaml not found. Using default internal configuration.")
        config = DEFAULT_CONFIG

    if 'smiles_col' not in config['dataset']:
        config['dataset']['smiles_col'] = SMILES_COL

    print(config)
    main_extract_vec(config)