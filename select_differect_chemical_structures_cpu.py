import os
import time
import argparse
from collections import defaultdict
from itertools import compress

import numpy as np
from ase import Atoms
from ase.io import read, write

from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel
from joblib import Parallel, delayed

from utils import compute_soap_descriptors, save_soap_to_hdf5, setup_total_logging, setup_logging

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

def compute_similarity(cand_soap, ref_soap, kernel_metric="laplacian"):
    """
    Function: Compute the similarity between candidate and reference SOAP descriptors using the specified kernel metric.
    Input:
        cand_soap: Candidate SOAP descriptor
        ref_soap: Reference SOAP descriptor
        kernel_metric: Kernel metric to use for similarity computation (default: laplacian)
    Output:
        Similarity score between candidate and reference SOAP descriptors
    """
    re = AverageKernel(metric=kernel_metric)
    return re.create(cand_soap, ref_soap)

def compare_and_update_structures(ref_structures, cand_structures, njobs=8, species=["H", "C", "O", "N"], r_cut=10.0, n_max=6, l_max=4, threshold=0.9, logger=None):
    """
    Function:
    Compare candidate structures with reference structures.
    Update the reference database one molecule by one molecule.

    Input:
        ref_structures: list of ase.Atoms objects, reference structures
        cand_structures: list of ase.Atoms objects, candidate structures
        njobs: int, number of jobs to run in parallel
        species: list of str, species to consider  
        r_cut: float, cutoff radius for SOAP calculation
        n_max: int, number of radial basis functions
        l_max: int, maximum degree of spherical harmonics
        threshold: float, similarity threshold for reducing candidate structures
        logger: logging.Logger object, logger for logging

    Output:
        ref_structures: list of ase.Atoms objects, updated reference structures
        soap_ref: list of soap_descriptors, SOAP descriptors for updated reference structures
    """
    round_num = 0
    logger.info(f"njobs: {njobs}, species: {species}, r_cut: {r_cut}, n_max: {n_max}, l_max: {l_max}, threshold: {threshold}")

    while True:
        round_num += 1

        if round_num == 1:
            # 初次计算全部的 SOAP 描述符
            # 这里还可以改进，先计算描述符或直接读入描述符
            soap_ref = compute_soap_descriptors(ref_structures, njobs, species, r_cut, n_max, l_max, logger)
            soap_cand = compute_soap_descriptors(cand_structures, njobs, species, r_cut, n_max, l_max, logger)

            # 如果 soap_ref 为空，则将 soap_cand 的第一个元素添加到 soap_ref 中
            if soap_ref == []:
                ref_structures.append(cand_structures[0])
                soap_ref.append(soap_cand[0])
                logger.info("Ref structure is empty, add the first Cand structure to Ref structure")

            # 并行计算 cand_structures 中每个结构与 ref_structures 中所有结构的相似度
            start_time = time.time()
            re_kernel_results = Parallel(n_jobs=njobs)(delayed(compute_similarity)(soap_cand[i:i+1], soap_ref) for i in range(len(soap_cand)))
            re_kernel = np.vstack(re_kernel_results)
            end_time = time.time()
            logger.info(f"Round {round_num}: Similarity computation completed in {end_time - start_time:.2f} seconds")

            # 选取 cand_structures 中每个结构与 ref_structures 中所有结构的最大相似度
            max_similarity_values = np.max(re_kernel, axis=1)
        
        else:
            # 并行计算 cand_structures 中每个结构与 ref_structures 新加入的结构的相似度
            # 变量覆盖释放内存空间
            ### 这里可以写成一个函数，便于复用
            start_time = time.time()
            re_kernel_results = Parallel(n_jobs=njobs)(delayed(compute_similarity)(soap_cand[i:i+1], [soap_ref[-1]]) for i in range(len(soap_cand)))
            re_kernel = np.vstack(re_kernel_results)
            end_time = time.time()
            logger.info(f"Round {round_num}: Similarity computation completed in {end_time - start_time:.2f} seconds")
            ### 这里可以写成一个函数，便于复用

            # 将原先 max_similarity_values 与新加入的点的堆叠
            max_similarity_values = np.max(np.column_stack((max_similarity_values, np.max(re_kernel, axis=1))), axis=1)


        # 删除 cand_structures 中与 ref_structures 中相似度高于 threshold 的所有结构
        # 更新 soap_cand, cand_structures, max_similarity_values
        old_cand_num = len(cand_structures)

        preserve_condition = max_similarity_values < threshold # 减少不必要的 round(5) 开销
        soap_cand = list(compress(soap_cand, preserve_condition)) # itertools.compress() 更高效
        cand_structures = list(compress(cand_structures, preserve_condition))
        max_similarity_values = max_similarity_values[preserve_condition] # np 布尔索引更高效
        
        new_cand_num = len(cand_structures)
        logger.info(f"Round {round_num}: Cand structures reduced from {old_cand_num} to {new_cand_num}")

        # 如果 cand_structures 中没有元素，则退出循环
        if new_cand_num == 0:
            break

        # 将 cand_structures 与 ref_structures 中最不相似的结构添加到 ref_structures 中
        min_max_similarity = np.min(max_similarity_values).round(5) # 减少不必要的 round(5) 开销
        min_max_similarity_index = np.argmin(max_similarity_values)
        ref_structures.append(cand_structures[min_max_similarity_index])
        soap_ref.append(soap_cand[min_max_similarity_index])
        logger.info(f"Round {round_num}: Added structure with min max similarity {min_max_similarity}.")
        logger.info(f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}")
        logger.info("---------")


    logger.info("No structures remaining in candidate list.")
    logger.info(f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}")
    logger.info("---------")
        
    return ref_structures, soap_ref


# 主程序
def main(ref_file, cand_file, njobs, r_cut, n_max, l_max, threshold):
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
    total_logger.info("---------")


    for i, formula in enumerate(cand_dict.keys()):
        # 如果 ref_dict 中没有该组，则会返回空列表，程序可以正常运行
        total_logger.info(f"Processing formula {i+1:>}/{formula_num:>}: {formula}")
        total_logger.info(f"Start Ref structures: {len(ref_dict[formula])}, Cand structures: {len(cand_dict[formula])}")

        logger = setup_logging(formula)  
        formula_start_time = time.time()
        logger.info('Log begin')
        logger.info(f"Processing formula: {formula}")

        # threshold 对于 species 很敏感，这可能是自动匹配 species 后可能出现的问题
        # species=list(set(cand_dict[formula][0].get_chemical_symbols()))
        updated_structures, updated_soap_list = compare_and_update_structures(ref_dict[formula], 
                                                                            cand_dict[formula], 
                                                                            njobs=njobs,
                                                                            # species=species,
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
    parser.add_argument('--njobs', type=int, default=8, help='Number of jobs for parallel processing')
    parser.add_argument('--r_cut', type=float, default=10.0, help='Cutoff radius for soap descriptor')
    parser.add_argument('--n_max', type=int, default=6, help='Number of radial basis functions')
    parser.add_argument('--l_max', type=int, default=4, help='Maximum degree of spherical harmonics')
    parser.add_argument('--threshold', type=float, default=0.9, help='Similarity threshold')

    args = parser.parse_args()

    main(args.ref, args.cand, args.njobs, args.r_cut, args.n_max, args.l_max, args.threshold)