import sys
import numpy as np
import time
from ase.io import read, write
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel

def compute_soap_descriptors(structures, species, r_cut, n_max, l_max):
    """
    为一组结构计算SOAP描述符。
    """
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
    )
    if len(structures) == 1:
        return [soap.create(structures)]
    return soap.create(structures)

def compare_and_update_structures(ref_structures, cand_structures, species, r_cut=10.0, n_max=2, l_max=2, threshold=0.99, log_file='update_log.txt'):
    """
    比较结构，将相似性最小的候选结构加入参考结构，并删除相似性大于阈值的候选结构。
    """
    with open(log_file, 'w') as log:
        round_num = 0

        while cand_structures:
            round_num += 1
            soap_ref = compute_soap_descriptors(ref_structures, species, r_cut, n_max, l_max)
            soap_cand = compute_soap_descriptors(cand_structures, species, r_cut, n_max, l_max)

            #re = AverageKernel(metric="linear")
            re = AverageKernel(metric="laplacian")
            re_kernel = re.create(soap_cand, soap_ref)
            max_similarity_values = np.max(re_kernel, axis=1)

            min_max_similarity = np.min([round(i,5) for i in max_similarity_values])
            if min_max_similarity >= threshold:
                log.write(f"Round {round_num}: No structure added. Min max similarity ({min_max_similarity}) >= threshold ({threshold}).\n")
                break

            min_max_similarity_index = np.argmin(max_similarity_values)
            ref_structures.append(cand_structures[min_max_similarity_index])

            print(f"Round {round_num}: Added structure with min max similarity {min_max_similarity}. "
                      f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}")

            log.write(f"Round {round_num}: Added structure with min max similarity {min_max_similarity}. "
                      f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}\n")

            cand_structures = [cand_structures[i] for i in range(len(cand_structures)) if round(max_similarity_values[i],5) < threshold]

    return ref_structures

if __name__ == "__main__":
    start_time = time.time()

    species = ["H", "O", "C", "N"]  # 根据需要调整元素列表
    ref_structures = read('rxn0148_init.xyz', index=':')
    # ref_structures = read('updated_ref_structures.xyz', index=':')
    cand_structures = read('rxn0148.xyz', index=':')

    updated_ref_structures = compare_and_update_structures(ref_structures, cand_structures, species, log_file='update_log.txt')

    write("updated_ref_structures.xyz", updated_ref_structures)
    print("Updated reference structures saved to 'updated_ref_structures.xyz'")

    elapsed_time = time.time() - start_time
    print(f"完成，总耗时: {elapsed_time:.2f} 秒")
