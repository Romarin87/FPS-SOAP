import argparse
import numpy as np
from collections import defaultdict
from ase.io import read
import h5py
from utils import time_logger

@time_logger
def calculate_distances(molecules, ignore_elements=[], only_pairs=[]):
    distance_dict = defaultdict(list)

    # 将 only_pairs 从字符串转换为元组
    only_pairs = [tuple(sorted(pair.split('-'))) for pair in only_pairs]

    for mol in molecules:
        symbols = mol.get_chemical_symbols()
        positions = mol.get_positions()
        
        for i in range(len(symbols)):
            # 忽略指定元素
            if symbols[i] in ignore_elements:
                continue
            for j in range(i + 1, len(symbols)):
                # 忽略指定元素
                if symbols[j] in ignore_elements:
                    continue
                
                element_pair = tuple(sorted([symbols[i], symbols[j]])) # 排序元素对，确保唯一性
                
                # 只计算指定的元素对
                if only_pairs and element_pair not in only_pairs:
                    continue
                
                distance = np.linalg.norm(positions[i] - positions[j])  # 计算距离
                distance_dict[element_pair].append(distance)
    
    # 将 defaultdict 转换为普通字典
    return {f"{pair[0]}-{pair[1]}": distances for pair, distances in distance_dict.items()}

def main():
    parser = argparse.ArgumentParser(description="Calculate distances between atoms.")
    parser.add_argument('--file', type=str, required=True, help='Input file in .xyz format')
    parser.add_argument('--ignore', type=str, default='', 
                        help='Comma-separated list of elements to ignore (e.g., H,O)')
    parser.add_argument('--only', type=str, default='', 
                        help='Comma-separated list of the only element pairs to calculate (e.g., C-C,O-C)')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file name')

    args = parser.parse_args()

    # 处理 ignore_elements 和 only_pairs
    ignore_elements = args.ignore.split(',') if args.ignore else []
    only_pairs = args.only.split(',') if args.only else []

    print(f"Begin to read {args.file}")
    molecules = read(args.file, index=':')
    print(f"Finished reading {args.file}")

    print(f"Begin to calculate distances")
    distances = calculate_distances(molecules, ignore_elements, only_pairs)
    print(f"Finished calculating distances")

    # 保存距离到 HDF5 文件
    with h5py.File(args.output, 'w') as hdf:
        for key, value in distances.items():
            hdf.create_dataset(key, data=value)
    print(f"Save distances to {args.output}")

if __name__ == "__main__":
    main()
