import os
import time
import argparse
from collections import defaultdict
from itertools import compress

import numpy as np
from ase import Atoms
from ase.io import read, write
import torch

from dscribe.descriptors import SOAP
from joblib import Parallel, delayed

from utils import compute_soap_descriptors, save_soap_to_hdf5, setup_total_logging, setup_logging

class AverageKernelMultiDevice:
    def __init__(self, metric, gamma, gpu_id=None):
        """
        Args:
            metric (str): The pairwise metric used for calculating the local similarity, only "laplacian" is supported now.
            gamma (float): Gamma parameter for Laplacian kernel. Use sklearn's default gamma.
            gpu_id (int): The GPU device ID to be used for computation.
        """
        self.metric = metric
        self.gamma = gamma
        self.gpu_id = gpu_id

    def get_pairwise_matrix(self, X, Y=None):
        """
        Computes the pairwise similarity of atomic environments using Laplacian kernel on GPU.
        
        Args:
            X (torch.Tensor): Feature vector for atoms in multiple structures (n_x, n_atoms_x, n_features).
            Y (torch.Tensor): Feature vector for atoms in multiple structures (n_y, n_atoms_y, n_features).
                              If None, the pairwise similarity is computed between the same structures in X.
        
        Returns:
            torch.Tensor: Tensor (n_x, n_y, n_atoms_x, n_atoms_y) representing the pairwise similarities between X and Y.
                          If Y is None, the returned matrix is of shape (n_x, n_atoms_x, n_atoms_x).
        """
        device = torch.device(f'cuda:{self.gpu_id}' if (torch.cuda.is_available() and self.gpu_id is not None) else 'cpu')
        X = X.to(dtype=torch.float32, device=device)  # Shape: (n_x, n_atoms_x, n_features)

        if self.metric == "laplacian":

            # Normalization
            if Y is None:
                Y = X  # Shape: (n_x, n_atoms_x, n_features)
                diff = torch.abs(X.unsqueeze(2) - Y.unsqueeze(1))  # Shape: (n_x, n_atoms_x, n_atoms_x, n_features)
                dist = torch.sum(diff, dim=-1)  # Shape: (n_x, n_atoms_x, n_atoms_x)
                K_ij = torch.exp(-self.gamma * dist) # Shape: (n_x, n_atoms_x, n_atoms_x)

            else:
                Y = Y.to(dtype=torch.float32, device=device) # Shape: (n_y, n_atoms_y, n_features)

                # Broadcast difference calculation: compute |X_i - Y_j| for all i, j pairs
                diff = torch.abs(X.unsqueeze(1).unsqueeze(3) - Y.unsqueeze(0).unsqueeze(2))  # Shape: (n_x, n_y, n_atoms_x, n_atoms_y, n_features)

                # Sum over the atoms dimension (dim=3 for X and dim=4 for Y)
                dist = torch.sum(diff, dim=-1)  # Shape: (n_x, n_y, n_atoms_x, n_atoms_y)

                # Sum over atoms (2nd and 3rd dims) to get the pairwise kernel value for each pair of molecules
                K_ij = torch.exp(-self.gamma * dist)  # Shape: (n_x, n_y, n_atoms_x, n_atoms_y)

        return K_ij

    def get_global_similarity(self, localkernel):
        """
        Computes the average global similarity between two structures.
        
        Args:
            localkernel (torch.Tensor): Tensor (n_x, n_y, n_atoms_x, n_atoms_y) representing the pairwise similarities between structures in X and Y.
        
        Returns:
            torch.Tensor: Tensor (n_x, n_y) representing the average similarity between the structures.
                          If normalization mode in get_pairwise_matrix(), the shape returned is tensor (n_x).
        """
        device = torch.device(f'cuda:{self.gpu_id}' if (torch.cuda.is_available() and self.gpu_id is not None) else 'cpu')
        localkernel = localkernel.clone().detach().to(dtype=torch.float32, device=device)
        
        # Average similarity across all atoms in both molecules
        K_ij = torch.mean(localkernel, dim=(-2, -1))  # Shape: (n_x, n_y) or (n_x)

        return K_ij

    def create(self, x, y=None):
        """
        Creates the kernel matrix based on the given lists of local features x and y.
    
        Args:
            x (iterable): A list of local feature arrays for each structure. Each element is a tensor of shape (n_atoms, n_features).
            y (iterable): An optional second list of features. 
                          If not specified, y is assumed to be the same as x, and the function computes self-similarity.

        Returns:
            torch.Tensor: A tensor representing the pairwise global similarity kernel K[i,j] between the given structures. 
                          Shape: (n_x, n_y) or (n_x), depending on whether y is provided.
        """
        # If y is None, compute self-similarity using x only
        if y is None:
            x_tensor = torch.stack([torch.tensor(i, dtype=torch.float32) for i in x])
            localkernel = self.get_pairwise_matrix(x_tensor)
            K_ij = self.get_global_similarity(localkernel)
            K_ij = torch.sqrt(K_ij)

        # If y is provided, compute pairwise similarity between x and y
        else:
            # Convert input features to tensors
            x_tensor = torch.stack([torch.tensor(i, dtype=torch.float32) for i in x])
            y_tensor = torch.stack([torch.tensor(i, dtype=torch.float32) for i in y])

            # Compute pairwise kernel between structures in x and y
            localkernel = self.get_pairwise_matrix(x_tensor, y_tensor)

            # Compute global similarity between structures in x and y
            K_ij = self.get_global_similarity(localkernel)

        return K_ij

def compute_soap_descriptors(structures, njobs, species, r_cut, n_max, l_max, logger):
    """
    Function: Compute SOAP descriptors for a list of structures
    Input:
        SOAP inputs
        logger: logger object
    Output:
        List of SOAP descriptors in numpy.ndarray format
    """
    start_time = time.time()
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max
    )

    try:
        soap_descriptors = soap.create(structures, n_jobs=njobs)
    except IndexError:
        # 说明 structures 为空
        soap_descriptors = []

    end_time = time.time()
    logger.info(f"SOAP descriptors computed in {end_time - start_time:.2f} seconds")

    # 由于此处调用本函数计算时，structures 中的结构默认是化学式相同的，因此返回的是 np.ndarray
    # 需要返回 list 类型，其中元素为 np.ndarray
    if type(soap_descriptors) == np.ndarray:
        return [i for i in soap_descriptors]
    
    # 原子数不相同时直接返回 list 即可
    return soap_descriptors

def compute_similarity_pytorch(cand_soap, ref_soap=None, kernel_metric="laplacian", gpu_id=None):
    """
    Computes the pairwise similarity between candidate and reference SOAP descriptors using laplacian kernel metric.
    """
    # 以 sci-kit learn 相同的方法计算 gamma 值
    gamma = 1.0 / cand_soap[0].shape[1]

    re = AverageKernelMultiDevice(metric=kernel_metric, gamma=gamma, gpu_id=gpu_id)
    return re.create(cand_soap, ref_soap)

def compare_and_update_structures(ref_structures, cand_structures, njobs=4, gpu=1, batch_size=50, species=["H", "C", "O", "N"], r_cut=10.0, n_max=6, l_max=4, threshold=0.9, logger=None):
    round_num = 0
    logger.info(f"njobs: {njobs}, gpu: {gpu}, batch_size: {batch_size}")
    logger.info(f"species: {species}, r_cut: {r_cut}, n_max: {n_max}, l_max: {l_max}, threshold: {threshold}")

    while True:
        round_num += 1

        # 初次计算
        if round_num == 1:

            # 先计算全部的 SOAP 描述符
            soap_ref = compute_soap_descriptors(ref_structures, njobs, species, r_cut, n_max, l_max, logger)
            soap_cand = compute_soap_descriptors(cand_structures, njobs, species, r_cut, n_max, l_max, logger)

            # 如果 soap_ref 为空，则将 soap_cand 的第一个元素添加到 soap_ref 中
            if soap_ref == []:
                ref_structures.append(cand_structures[0])
                soap_ref.append(soap_cand[0])
                logger.info("Ref structure is empty, add the first Cand structure to Ref structure")

            start_time = time.time()

            # 支持 GPU 计算
            # 目前仅支持单 GPU 计算
            if gpu:
                # 分 batch_size 批次计算 cand 中每个结构与 ref 中所有结构的相似度
                re_kernel_results = Parallel(n_jobs=gpu)(
                    delayed(compute_similarity_pytorch)(soap_cand[i:i+batch_size], soap_ref, gpu_id=(i//batch_size)%gpu) 
                    for i in range(0, len(soap_cand), batch_size))

                # 分别计算 cand 与 ref 中结构的自我相似度，用于正则化最终的相似度结果到 [0, 1]
                soap_cand_self = Parallel(n_jobs=gpu)(
                    delayed(compute_similarity_pytorch)(soap_cand[i:i+batch_size], gpu_id=(i//batch_size)%gpu) 
                    for i in range(0, len(soap_cand), batch_size))

                soap_ref_self = Parallel(n_jobs=gpu)(
                    delayed(compute_similarity_pytorch)(soap_ref[i:i+batch_size], gpu_id=(i//batch_size)%gpu) 
                    for i in range(0, len(soap_ref), batch_size))

            # 支持多核 CPU 并行运算
            # 但会出现内存泄漏导致效率反而不如单核运算
            else:
                re_kernel_results = Parallel(n_jobs=njobs)(
                    delayed(compute_similarity_pytorch)(soap_cand[i:i+1], soap_ref) 
                    for i in range(0, len(soap_cand)))

                soap_cand_self = Parallel(n_jobs=njobs)(
                    delayed(compute_similarity_pytorch)(soap_cand[i:i+1]) 
                    for i in range(0, len(soap_cand)))

                soap_ref_self = Parallel(n_jobs=njobs)(
                    delayed(compute_similarity_pytorch)(soap_ref[i:i+1]) 
                    for i in range(0, len(soap_ref)))

            # 合并批次/并行计算的结果
            re_kernel = torch.cat(re_kernel_results, dim=0)
            soap_cand_self = torch.cat(soap_cand_self, dim=0)
            soap_ref_self = torch.cat(soap_ref_self, dim=0)

            # 正则化相似度矩阵
            re_kernel /= torch.outer(soap_cand_self, soap_ref_self)

            # 将相似度矩阵移动到 CPU 以避免 I/O 造成的计算速度瓶颈
            re_kernel = re_kernel.cpu()

            end_time = time.time()
            logger.info(f"Round {round_num}: Similarity computation and self-similarity completed in {end_time - start_time:.2f} seconds")

            # 选取 cand_structures 中每个结构与 ref_structures 中所有结构的最大相似度
            max_similarity_values, _ = torch.max(re_kernel, dim=1)

        # 非初次计算
        else:
            start_time = time.time()

            if gpu:
                re_kernel_results = Parallel(n_jobs=gpu)(
                    delayed(compute_similarity_pytorch)(soap_cand[i:i+batch_size], [soap_ref[-1]], gpu_id=(i//batch_size)%gpu) 
                    for i in range(0, len(soap_cand), batch_size))
            else:
                re_kernel_results = Parallel(n_jobs=njobs)(
                    delayed(compute_similarity_pytorch)(soap_cand[i:i+1], [soap_ref[-1]]) 
                    for i in range(0, len(soap_cand)))
                                
            re_kernel = torch.cat(re_kernel_results, dim=0)
            re_kernel /= torch.outer(soap_cand_self, soap_ref_self[-1].unsqueeze(0))
            re_kernel = re_kernel.cpu()

            end_time = time.time()
            logger.info(f"Round {round_num}: Similarity computation completed in {end_time - start_time:.2f} seconds")

            # 将原先 max_similarity_values 与新加入的 ref 计算的 max_similarity_value 合并
            max_similarity_values, _ = torch.max(
                                            torch.cat((max_similarity_values.view(-1, 1), 
                                                       torch.max(re_kernel, dim=1)[0].view(-1, 1)), 
                                                       dim=1), 
                                            dim=1)
            
        # 删除 cand_structures 中与 ref_structures 中相似度高于 threshold 的所有结构
        # 更新 soap_cand, cand_structures, max_similarity_values, soap_cand_self
        old_cand_num = len(cand_structures)

        preserve_condition = max_similarity_values < threshold
        soap_cand = list(compress(soap_cand, preserve_condition)) # itertools.compress() 更高效
        cand_structures = list(compress(cand_structures, preserve_condition))
        max_similarity_values = max_similarity_values[preserve_condition] # 布尔索引更高效
        soap_cand_self = soap_cand_self[preserve_condition]

        new_cand_num = len(cand_structures)
        logger.info(f"Round {round_num}: Cand structures reduced from {old_cand_num} to {new_cand_num}")

        # 如果 cand_structures 中没有元素，说明筛选完毕，退出循环
        if new_cand_num == 0:
            break
        
        # 将 cand_structures 与 ref_structures 中最不相似的结构添加到 ref_structures 中
        # 包含结构，SOAP 描述符，以及用于正则化的自我相似度
        min_max_similarity = torch.min(max_similarity_values).item()
        min_max_similarity_index = torch.argmin(max_similarity_values).item()
        ref_structures.append(cand_structures[min_max_similarity_index])
        soap_ref.append(soap_cand[min_max_similarity_index])
        soap_ref_self = torch.cat((soap_ref_self, soap_cand_self[min_max_similarity_index].unsqueeze(0)))

        logger.info(f"Round {round_num}: Added structure with min max similarity {min_max_similarity:.5f}.")
        logger.info(f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}")
        logger.info("---------")

    logger.info("No structures remaining in candidate list.")
    logger.info(f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}")
    logger.info("---------")

    return ref_structures, soap_ref


# 主程序
def main(ref_file, cand_file, njobs, gpu, batch_size, r_cut, n_max, l_max, threshold):
    total_logger = setup_total_logging()
    total_logger.info('Total Log begin')
    start_time = time.time()

    # 读取数据
    if ref_file == '':
        ref_structures = []
    else:
        ref_structures = read(ref_file, index=':')
    cand_structures = read(cand_file, index=':')

    # 根据 chemical_formula 分组
    ref_dict = defaultdict(list)
    cand_dict = defaultdict(list)

    for structure in ref_structures:
        formula = structure.get_chemical_formula()
        ref_dict[formula].append(structure)

    for structure in cand_structures:
        formula = structure.get_chemical_formula()
        cand_dict[formula].append(structure)

    formula_num = len(cand_dict.keys())
    total_logger.info(f"There are {formula_num} formulas to process.")

    # 确认 species 中所含有的元素类型
    species = set()
    for key in cand_dict:
        species.update(cand_dict[key][0].get_chemical_symbols())
    for key in ref_dict:
        try:
            species.update(cand_dict[key][0].get_chemical_symbols())
        except:
            continue
    species = list(species)
    total_logger.info(f"Species: {species}")
    total_logger.info("---------")

    for i, formula in enumerate(cand_dict.keys()):
        # 如果 ref_dict 中没有该组，则会返回空列表，程序可以正常运行
        total_logger.info(f"Processing formula {i+1:>}/{formula_num:>}: {formula}")
        total_logger.info(f"Start Ref structures: {len(ref_dict[formula])}, Cand structures: {len(cand_dict[formula])}")

        logger = setup_logging(formula)  
        formula_start_time = time.time()
        logger.info('Log begin')
        logger.info(f"Processing formula: {formula}")

        # 由于后续相似度是经过正则化到 [0, 1] 的，因此不必对每一个反应自动匹配 species，直接取所有元素的并集即可
        # species=list(set(cand_dict[formula][0].get_chemical_symbols()))
        updated_structures, updated_soap_list = compare_and_update_structures(ref_dict[formula], 
                                                                              cand_dict[formula], 
                                                                              njobs=njobs,
                                                                              gpu=gpu,
                                                                              batch_size=batch_size,                                                                                  
                                                                              species=species,
                                                                              r_cut=r_cut,
                                                                              n_max=n_max,
                                                                              l_max=l_max,
                                                                              threshold=threshold,
                                                                              logger=logger)
        
        # 逐一保存更新后的参考结构
        write(os.path.join(formula, f"updated_ref_structures_{formula}.xyz"), updated_structures)
        logger.info(f"Updated reference structures saved to '{formula}/updated_ref_structures_{formula}.xyz'")
        
        # 保存更新后的结构和 SOAP 到 HDF5
        soap_dict = defaultdict(list)
        for i in range(len(updated_soap_list)):
            soap_result = updated_soap_list[i]
            soap_dict[formula].append(soap_result)
        save_soap_to_hdf5(soap_dict, os.path.join(formula, f"updated_ref_soap_descriptors_{formula}.h5"))
        logger.info(f"SOAP descriptors saved to '{formula}/updated_ref_soap_descriptors_{formula}.h5'")

        formula_end_time = time.time()
        logger.info(f"Done! Total time elapsed: {formula_end_time - formula_start_time:.2f} seconds")
        logger.info('Log end')

        total_logger.info(f"End Ref structures: {len(updated_structures)}")
        total_logger.info(f"Done! Total time elapsed: {formula_end_time - formula_start_time:.2f} seconds")
        total_logger.info("---------")

    end_time = time.time()
    total_logger.info('All reactions processed successfully!')
    total_logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    total_logger.info('Total Log end')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select different chemical structures.')
    parser.add_argument('--ref', type=str, default='', help='Reference XYZ file')
    parser.add_argument('--cand', type=str, required=True, help='Candidate XYZ file')
    parser.add_argument('--njobs', type=int, default=1, help='Number of jobs for CPU parallel processing')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs for GPU parallel processing')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for GPU parallel processing')
    parser.add_argument('--r_cut', type=float, default=10.0, help='Cutoff radius for soap descriptor')
    parser.add_argument('--n_max', type=int, default=6, help='Number of radial basis functions')
    parser.add_argument('--l_max', type=int, default=4, help='Maximum degree of spherical harmonics')
    parser.add_argument('--threshold', type=float, default=0.9, help='Similarity threshold')

    args = parser.parse_args()

    main(args.ref, args.cand, args.njobs, args.gpu, args.batch_size, args.r_cut, args.n_max, args.l_max, args.threshold)
