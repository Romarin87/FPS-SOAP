import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
# from matplotlib.ticker import LogLocator

def draw_bond_distributions(distances_ls, labels, output, distance_range=(1, 5)):
    print('Begin to draw bond distributions')

    # 设置颜色
    color = ['#93bede', '#58a964', '#d76a69', '#f1c40f', '#e67e22', '#9b59b6']

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.sans-serif'] = ['Arial']

    # 获取所有键
    keys_union = set()
    for distances in distances_ls:
        keys_union.update(distances.keys())
    key_ls = sorted(list(keys_union))

    # 创建子图，动态调整子图的行数和列数，每行一个子图
    num_rows = len(key_ls)
    num_cols = 1 
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 1.5 * num_rows),
                            sharex='col')

    # 展平 axs 数组以方便索引
    axs = axs.flatten()

    for i in range(num_rows * num_cols):
        key = key_ls[i]
        for color_index, data in enumerate(distances_ls):
            try:
                # 计算直方图频次数据
                counts, bins = np.histogram(data[key], range=distance_range, bins=100)
            except:
                continue

            # 进行 log10 处理，并区分 0 与 1
            counts_log = []
            for x in counts:
                if x == 0:
                    counts_log.append(0)
                elif x == 1:
                    counts_log.append(np.log10(1.5))
                else:
                    counts_log.append(np.log10(x)) 

            # 绘制直方图
            axs[i].bar(bins[:-1], counts_log, width=np.diff(bins),
                       color=color[color_index % len(color)], alpha=0.7, label=key, align='edge')

        # 获取 y 轴的默认长度
        y_length = axs[i].get_ylim()[1]
        axs[i].set_yticks([np.round(y_length * 0.25 * num, 2) for num in range(5)])
        # # 使用 LogLocator 设置 y 轴刻度
        # axs[i].set_yscale('log')
        # axs[i].yaxis.set_major_locator(LogLocator(base=10.0, numticks=9))

        if i == 0:  # 第一行
            axs[i].set_title('Distribution of X-X length', loc='left', pad=10)
            axs[i].text(distance_range[0], y_length * 0.75, key)
        else:
            axs[i].text(distance_range[0], y_length * 0.75, key)

        if i == num_rows - 1:  # 最后一行
            axs[i].set_xlabel('Distance (Å)')
            axs[i].set_ylabel(' ') # 占位，以显示全局 y 轴标签

    # 添加全局 y 轴标签
    fig.text(0.016, 0.5, 'Log10 Frequency', va='center', ha='center', rotation='vertical')

    # 创建统一的 legend
    handles, _ = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',
            #    bbox_to_anchor=(0.9, 0.925),
               bbox_transform=fig.transFigure,
               ncol=3)

    plt.tight_layout()
    plt.savefig(f'{output}')
    print(f'Finish drawing bond distributions {output}')


def main():
    parser = argparse.ArgumentParser(description="Draw bond distribution graph.")
    # 尽量把认为分布频次更大的数据集放在前面
    parser.add_argument('--files', nargs='+', help='List of bond distribution filenames', required=True)
    parser.add_argument('--output', type=str, help='String of bond distribution output graph name', required=True)
    parser.add_argument('--range', type=str, default='1,5', 
                        help='Comma-separated list of distance ranges in angstrom (e.g., 1,5)', required=False)

    args = parser.parse_args()

    # 存储每个文件的距离数据
    distances_ls = []

    for file in args.files:
        distances = {}
        with h5py.File(file, 'r') as hdf:
            for key in hdf.keys():
                distances[key] = hdf[key][:]  # 读取每个键的数据
        distances_ls.append(distances)  # 将每个文件的距离字典存入列表

    # 处理范围参数
    distance_range = tuple(map(float, args.range.split(',')))

    draw_bond_distributions(distances_ls, args.files, args.output, distance_range)

if __name__ == "__main__":
    main()

