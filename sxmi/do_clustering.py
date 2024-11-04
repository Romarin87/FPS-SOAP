import argparse
import h5py
import numpy as np
import time
from functools import wraps
from sklearn.manifold import TSNE
import umap
from functions import time_logger

@time_logger
def do_tSNE(tSNE_params, flattened_matrices):
    print('Begin to do t-SNE')
    tsne = TSNE(**tSNE_params)
    reduced_data = tsne.fit_transform(flattened_matrices)

    with h5py.File('t-SNE.h5', 'w') as hdf:
        hdf.create_dataset('tsne_result', data=reduced_data)
    print('Finish doing t-SNE')

@time_logger
def do_UMAP(UMAP_params, flattened_matrices):
    print('Begin to do UMAP')
    umap_model = umap.UMAP(**UMAP_params)
    reduced_data = umap_model.fit_transform(flattened_matrices)

    with h5py.File('UMAP.h5', 'w') as hdf:
        hdf.create_dataset('umap_result', data=reduced_data)
    print('Finish doing UMAP')

def load_h5_datasets(filenames, key='coulomb_matrices'):
    data_list = []
    for filename in filenames:
        with h5py.File(filename, 'r') as hdf:
            data = hdf[key][:]  # 读取数据
            data_list.append(data)
    return data_list

def main():
    parser = argparse.ArgumentParser(description="Do t-SNE and UMAP.")
    # 深色在前，浅色在后
    parser.add_argument('--files', nargs='+', help='List of filenames', required=True)
    parser.add_argument('--tsne', type=str, default=None, 
                        help='Custom parameters for t-SNE (e.g., "n_components=2,perplexity=30")')
    parser.add_argument('--umap', type=str, default=None, 
                        help='Custom parameters for UMAP (e.g., "n_components=2,n_neighbors=10")')
    args = parser.parse_args()
    
    # 检查是否至少提供了 tsne 或 umap
    if not (args.tsne or args.umap):
        parser.error("At least one of --tsne or --umap must be provided.")

    print('Begin to concatenate data')
    
    # 从命令行参数中获取文件列表
    filenames = args.files
    data_list = load_h5_datasets(filenames)

    # 将所有数据合并
    flattened_matrices = np.concatenate(data_list, axis=0)

    # 创建标签
    labels = np.concatenate(
        [np.full(len(data), i + 1) for i, data in enumerate(data_list)],
        axis=0
    )

    # 保存标签为 HDF5 文件
    with h5py.File('labels.h5', 'w') as hdf:
        hdf.create_dataset('legends', data=filenames)
        hdf.create_dataset('labels', data=labels)

    print('Finish concatenating data')

    # 进行 t-SNE 降维
    tSNE_params = {}
    if args.tsne:
        tSNE_params = dict(param.split('=') for param in args.tsne.split(','))

        # 将字符串类型的参数转换为合适的类型
        for key in tSNE_params:
            if key in ['n_components', 'max_iter', 'n_iter_without_progress', 'verbose', 'random_state']:
                tSNE_params[key] = int(tSNE_params[key])  # 转换为整数
            elif key in ['perplexity', 'early_exaggeration', 'learning_rate', 'min_grad_norm']:
                tSNE_params[key] = float(tSNE_params[key])  # 转换为浮点数

        do_tSNE(tSNE_params, flattened_matrices)

    # 进行 UMAP 降维
    UMAP_params = {}
    if args.umap:
        UMAP_params = dict(param.split('=') for param in args.umap.split(','))

        # 将字符串类型的参数转换为合适的类型
        for key in UMAP_params:
            if key in ['n_neighbors', 'n_components', 'random_state']:
                UMAP_params[key] = int(UMAP_params[key])
            elif key in ['learning_rate', 'local_connectivity', 'min_dist']:
                UMAP_params[key] = float(UMAP_params[key])
            elif key in ['verbose']:
                UMAP_params[key] = (UMAP_params[key] in ['True', 'true', '1'])
            
        do_UMAP(UMAP_params, flattened_matrices)

if __name__ == "__main__":
    main()