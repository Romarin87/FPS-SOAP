import argparse
from ase.io import read
from dscribe.descriptors import CoulombMatrix
import h5py
from functions import time_logger

@time_logger
def calculate_coulomb_matrices(molecules, max_n_atoms, n_jobs):
    cm = CoulombMatrix(n_atoms_max=max_n_atoms)
    return cm.create(molecules, n_jobs=n_jobs)

def main():
    parser = argparse.ArgumentParser(description="Calculate Coulomb matrices from atomic structures.")
    parser.add_argument('--file', type=str, required=True, help='Input file in .xyz format')
    parser.add_argument('--output', type=str, required=True, help='Output file name for the Coulomb matrices (HDF5 format)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs to use for computation')

    args = parser.parse_args()

    print(f"Begin to read {args.file}")
    molecules = read(args.file, index=':')
    print(f"Finished reading {args.file}")

    # 获取原子数的最大值
    max_n_atoms = max(len(mol) for mol in molecules)
    print(f"Maximum number of atoms in a molecule: {max_n_atoms}")

    # 计算库伦矩阵
    print("Begin to calculate Coulomb matrices")
    coulomb_matrices = calculate_coulomb_matrices(molecules, max_n_atoms, args.n_jobs)
    print("Finished calculating Coulomb matrices")

    # 保存库伦矩阵到 HDF5 文件
    with h5py.File(args.output, 'w') as hdf:
        hdf.create_dataset('coulomb_matrices', data=coulomb_matrices)
    print(f"Save Coulomb matrices to {args.output}")

if __name__ == "__main__":
    main()
