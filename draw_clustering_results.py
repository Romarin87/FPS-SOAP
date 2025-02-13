import matplotlib.pyplot as plt
import argparse
import h5py

def draw_clustering_results(reduced_data, labels, legends, name):
    print('Begin to draw clustering results')
    plt.figure(figsize=(8, 8), dpi=500)

    # 设置颜色列表
    colors = [
    '#440154',  # 黄色
    '#FDE724',  # 紫色
    '#55a868',  # 绿色
    '#c44e52',  # 红色
    '#4c72b0',  # 蓝色
    '#7b9a2e',  # 棕色
    '#a058b2',  # 粉色
    '#c6c6c6'   # 灰色
    '#dd8452',  # 橙色
]

    # 从最后一个标签开始绘制
    for i in range(len(legends) - 1, -1, -1):
        mask = labels == (i + 1)  # 标签从 1 开始
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                    color=colors[i % len(colors)], alpha=0.7, s=1, label=legends[i])

    # 添加图例
    plt.legend()
    plt.title(f'{name} Visualization of Molecular Structures with Coulomb Matrices')
    plt.xlabel(f'{name} Component 1')
    plt.ylabel(f'{name} Component 2')
    plt.savefig(f'{name}.png')
    print(f'Finish drawing clustering results {name}.png')

def main():
    parser = argparse.ArgumentParser(description="Draw clustering results.")
    # 创建互斥的参数组
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--tsne', type=str, default=None, help='Path to the t-SNE HDF5 file')
    group.add_argument('--umap', type=str, default=None, help='Path to the UMAP HDF5 file')
    parser.add_argument('--labels', type=str, help='Path to the labels HDF5 file', required=True)
    args = parser.parse_args()

    if args.tsne:
        name = "t-SNE"
        with h5py.File(args.tsne, 'r') as tsne_h5:
            reduced_data = tsne_h5['tsne_result'][:]  # 读取 t-SNE 数据
    else:
        name = "UMAP"
        with h5py.File(args.umap, 'r') as umap_h5:
            reduced_data = umap_h5['umap_result'][:]  # 读取 UMAP 数据

    with h5py.File(args.labels, 'r') as label_h5:
        labels = label_h5['labels'][:]  # 读取标签数据
        legends = label_h5['legends'][:]  # 读取图例数据
        legends = [legend.decode('utf-8') for legend in legends]
    
    draw_clustering_results(reduced_data, labels, legends, name)

if __name__ == "__main__":
    main()
