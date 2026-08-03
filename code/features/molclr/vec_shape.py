import numpy as np

# --- 修改为您要检查的文件名 ---
file_path = 'extracted_molclr_vectors_data_0.npy'
# -----------------------------

try:
    # 加载 .npy 文件
    data = np.load(file_path)

    # 打印核心信息：shape (形状)
    # 这会显示 (样本数, 特征维度)
    print(f"文件路径: {file_path}")
    print(f"数据维度 (Shape): {data.shape}")

    # 打印其他有用信息
    print(f"数据类型 (Dtype): {data.dtype}")
    print(f"总共有几维 (Ndim): {data.ndim}") # 2D, 3D...
    print(f"元素总数: {data.size}")

    # (可选项) 如果您想查看前几个向量长什么样
    # print("\n前 5 个向量的预览:")
    # print(data[:5])

except FileNotFoundError:
    print(f"错误: 找不到文件 '{file_path}'")
except Exception as e:
    print(f"加载文件时出错: {e}")