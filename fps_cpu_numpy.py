import argparse
import os
import time
from datetime import datetime
from collections import defaultdict
from itertools import compress

import numpy as np
import numba
from ase import Atoms
from ase.io import read, write
from dscribe.descriptors import SOAP
from numba import njit, prange
import psutil

from utils import save_soap_to_hdf5, setup_logging, setup_total_logging

@njit(fastmath=True, cache=True, parallel=True)
def laplacian_kernel_numba(X, Y=None, gamma=1.0):
    """
    Computes the Laplacian kernel matrix between two sets of atomic environments using Numba for acceleration.
    
    The Laplacian kernel is defined as:
    K(x, y) = exp(-γ * ||x - y||₁)
    where ||x - y||₁ is the L1 (Manhattan) distance between feature vectors x and y.

    Parameters:
    -----------
    X : numpy.ndarray, shape (n_x, n_atoms_x, n_features)
    Y : numpy.ndarray, shape (n_y, n_atoms_y, n_features). If None, Y is set to X.
    
    gamma : float, optional (default=1.0)
        Kernel coefficient controlling the width of the Gaussian.

    Returns:
    --------
    numpy.ndarray, shape (n_x, n_y, n_atoms_x, n_atoms_y)
        Laplacian kernel matrix between all pairs of atoms in all environments.
    """

    # Handle Y=None case by setting Y to X (for self-similarity calculation)
    if Y is None:
        Y = X
    
    # Get dimensions of input arrays
    n_x, n_atoms_x, n_features = X.shape
    n_y, n_atoms_y, _ = Y.shape
    
    # Initialize distance matrix with float32 precision for memory efficiency
    dist = np.zeros((n_x, n_y, n_atoms_x, n_atoms_y), dtype=np.float32)
    
    # Compute L1 (Manhattan) distance between all pairs of atomic feature vectors
    for i in range(n_x):
        for j in range(n_y):
            for k in prange(n_atoms_x):  # Parallel loop
                for l in range(n_atoms_y): 
                    # Compute L1 distance between feature vectors of atom k in X and atom l in Y
                    d = 0.0
                    for m in range(n_features):  # Sum over feature dimensions
                        d += abs(X[i, k, m] - Y[j, l, m])
                    dist[i, j, k, l] = d  # Store distance
    
    # Apply Laplacian kernel: exp(-γ * distance)
    return np.exp(-gamma * dist)


class AverageLaplacianKernel():
    def __init__(self, gamma):
        """
        Args:
            metric (str): The pairwise metric used for calculating the local similarity, only "laplacian" is supported now.
            gamma (float): Gamma parameter for Laplacian kernel. Use sklearn's default gamma.
        """
        self.gamma = gamma

    def get_pairwise_matrix(self, X, Y=None):
        """
        Computes the pairwise similarity of atomic environments using Laplacian kernel with NumPy (CPU parallel).
        
        Args:
            X (np.ndarray): Feature vector for atoms in multiple structures (n_x, n_atoms_x, n_features).
            Y (np.ndarray): Feature vector for atoms in multiple structures (n_y, n_atoms_y, n_features).
                            If None, the pairwise similarity is computed between the same structures in X.
        
        Returns:
            np.ndarray: Array (n_x, n_y, n_atoms_x, n_atoms_y) representing the pairwise similarities between X and Y.
        """

        # Numba accelerated version
        return laplacian_kernel_numba(X, Y, self.gamma)

        # Numpy version
        # if self.metric == "laplacian":
        #     # Normalization
        #     if Y is None:
        #         Y = X  # Shape: (n_x, n_atoms_x, n_features)
        #         diff = np.abs(X[:, np.newaxis, :, :] - Y[:, :, np.newaxis, :])  # Shape: (n_x, n_atoms_x, n_atoms_x, n_features)
        #         dist = np.sum(diff, axis=-1)  # Shape: (n_x, n_atoms_x, n_atoms_x)
        #         K_ij = np.exp(-self.gamma * dist)  # Shape: (n_x, n_atoms_x, n_atoms_x)
        #     else:
        #         Y = Y.astype(np.float32)  # Shape: (n_y, n_atoms_y, n_features)
                
        #         # Broadcast difference calculation: compute |X_i - Y_j| for all i, j pairs
        #         diff = np.abs(X[:, np.newaxis, :, np.newaxis, :] - Y[np.newaxis, :, np.newaxis, :, :])  # Shape: (n_x, n_y, n_atoms_x, n_atoms_y, n_features)
                
        #         # Sum over the features dimension
        #         dist = np.sum(diff, axis=-1)  # Shape: (n_x, n_y, n_atoms_x, n_atoms_y)
                
        #         # Compute Laplacian kernel
        #         K_ij = np.exp(-self.gamma * dist)  # Shape: (n_x, n_y, n_atoms_x, n_atoms_y)
        
        # return K_ij

    def get_global_similarity(self, localkernel):
        """
        Computes the average global similarity between two structures using NumPy.
        
        Args:
            localkernel (np.ndarray): Array (n_x, n_y, n_atoms_x, n_atoms_y) representing the pairwise similarities between structures in X and Y.
        
        Returns:
            np.ndarray: Array (n_x, n_y) representing the average similarity between the structures.
        """
    
        # Average similarity across all atoms in both molecules
        K_ij = np.mean(localkernel, axis=(-2, -1))  # Shape: (n_x, n_y)

        return K_ij

    def create(self, x, y=None):
        """
        Creates the kernel matrix based on the given lists of local features x and y using NumPy.
        
        Args:
            x (iterable): A list of local feature arrays for each structure. Each element is an array of shape (n_atoms, n_features).
            y (iterable): An optional second list of features. 
                          If not specified, y is assumed to be the same as x, and the function computes self-similarity.

        Returns:
            np.ndarray: An array representing the pairwise global similarity kernel K[i,j] between the given structures. 
                        Shape: (n_x, n_y). If y is None, n_y = 1.
        """

        # If y is None, compute self-similarity using x only
        if y is None:
            localkernel = self.get_pairwise_matrix(x)
            K_ij = self.get_global_similarity(localkernel)
            K_ij = np.sqrt(K_ij)

        # If y is provided, compute pairwise similarity between x and y
        else:
            # Compute pairwise kernel between structures in x and y
            localkernel = self.get_pairwise_matrix(x, y)

            # Compute global similarity between structures in x and y
            K_ij = self.get_global_similarity(localkernel)

        return K_ij

def compute_soap_descriptors(structures, njobs, species, r_cut, n_max, l_max):
    """
    Function: Compute SOAP descriptors for a list of structures
    Input:
        SOAP inputs
        logger: logger object
    Output:
        List of SOAP descriptors in numpy.ndarray format
    """
    if not structures:
        return np.array([], dtype=np.float32)
    
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max
    )

    soap_descriptors = soap.create(structures, n_jobs=njobs)

    # For shape compatibility
    if len(structures) == 1:
        return np.array([i for i in [soap_descriptors]], dtype=np.float32)

    return np.array([i for i in soap_descriptors], dtype=np.float32)

def compute_similarity_numpy(cand_soap, ref_soap=None, gamma=1.0):
    """
    Computes the pairwise similarity between candidate and reference SOAP descriptors using laplacian kernel metric.
    """

    re = AverageLaplacianKernel(gamma=gamma)
    return re.create(cand_soap, ref_soap)

def compare_and_update_structures(ref_structures, cand_structures, n_jobs=None, batch_size=50, species=["H", "C", "O", "N"], r_cut=10.0, n_max=6, l_max=4, 
                                  threshold=0.9, max_fps_rounds=None, save_soap=False, logger=None):
    round_num = 0

    if n_jobs is None:
        n_jobs = psutil.cpu_count(logical=False)
    numba.set_num_threads(n_jobs)

    logger.info(f"n_jobs: {n_jobs}, batch_size: {batch_size}, max_fps_rounds: {max_fps_rounds}")
    logger.info(f"species: {species}, r_cut: {r_cut}, n_max: {n_max}, l_max: {l_max}, threshold: {threshold}")
    logger.info(f"Save SOAP descriptors: {save_soap}")

    while True:
        round_num += 1

        # Round 1
        if round_num == 1:
            
            # Compute SOAP descriptors
            logger.info("Computing SOAP descriptors...")
            start_time = time.time()
            soap_ref = compute_soap_descriptors(ref_structures, n_jobs, species, r_cut, n_max, l_max)
            soap_cand = compute_soap_descriptors(cand_structures, n_jobs, species, r_cut, n_max, l_max)
            end_time = time.time()
            logger.info(f"SOAP descriptors computed in {end_time - start_time:.2f}")
            logger.info(f"SOAP of ref in shape: {soap_ref.shape}, SOAP of cand in shape: {soap_cand.shape}")

            # If soap_ref is empty, add the first element of soap_cand to soap_ref
            if soap_ref.size == 0:
                ref_structures.append(cand_structures[0])
                soap_ref = soap_cand[0:1]
                logger.info("Ref structure is empty, add the first Cand structure to Ref structure")
            
            # Calculate gamma value, using the same method as sci-kit learn
            gamma = 1.0 / soap_cand.shape[-1] # SOAP descriptor dimension

            # Numba optimization, support multi-core CPU parallel computation
            start_time = time.time()
            re_kernel_results = [compute_similarity_numpy(soap_cand[i:i+1], soap_ref, gamma=gamma)
                for i in range(len(soap_cand))]
            soap_cand_self = [compute_similarity_numpy(soap_cand[i:i+1], gamma=gamma)
                for i in range(len(soap_cand))]
            soap_ref_self = [compute_similarity_numpy(soap_ref[i:i+1], gamma=gamma) 
                for i in range(len(soap_ref))]
            end_time = time.time()
            logger.info(f"Round {round_num}: Similarity computation and self-similarity completed in {end_time - start_time:.2f} seconds")
            
            # Concatenate the results of each batch
            re_kernel = np.concatenate(re_kernel_results, axis=0)
            soap_cand_self = np.concatenate(soap_cand_self, axis=0)
            soap_ref_self = np.concatenate(soap_ref_self, axis=0)

            # logger.info(re_kernel.shape)
            # logger.info(soap_cand_self.shape)
            # logger.info(soap_ref_self.shape)

            # Normalize the similarity matrix
            re_kernel /= np.outer(soap_cand_self, soap_ref_self)

            # Calculate the maximum similarity for each candidate structure with respect to all reference structures
            max_similarity_values = np.max(re_kernel, axis=1)
            logger.info(f"max_similarity_values shape: {max_similarity_values.shape}")

        # Round 2 and afterwards
        else:

            # Key bottleneck of calculation
            # Numba optimization, support multi-core CPU parallel computation
            start_time = time.time()
            batch_size = 50
            re_kernel_results = [compute_similarity_numpy(soap_cand[i:i+batch_size], soap_ref[-1:], gamma=gamma)
                for i in range(0, len(soap_cand), batch_size)]
            end_time = time.time()
            logger.info(f"Round {round_num}: Similarity computation completed in {end_time - start_time:.2f} seconds")

            # Concatenate and Normalize
            re_kernel = np.concatenate(re_kernel_results, axis=0)
            re_kernel /= np.outer(soap_cand_self, soap_ref_self[-1])
            
            # Update the maximum similarity values
            max_similarity_values = np.maximum(
                max_similarity_values,
                np.max(re_kernel, axis=1)
            )
        
        # Delete cand_structures that have a similarity value greater than the threshold
        # Update soap_cand, cand_structures, max_similarity_values, soap_cand_self
        old_cand_num = len(cand_structures)

        preserve_condition = max_similarity_values < threshold
        soap_cand = soap_cand[preserve_condition]
        cand_structures = list(compress(cand_structures, preserve_condition))
        max_similarity_values = max_similarity_values[preserve_condition]
        soap_cand_self = soap_cand_self[preserve_condition]

        new_cand_num = len(cand_structures)
        logger.info(f"Round {round_num}: Cand structures reduced from {old_cand_num} to {new_cand_num}")

        # If cand_structures is empty, meaning the fps is finished, break the loop
        if new_cand_num == 0:
            logger.info("No structures remaining in candidate list.")
            logger.info(f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}")
            break
        
        # Update the reference structures with the most unsimilar structure from cand_structures
        min_max_similarity = np.min(max_similarity_values)
        min_max_similarity_index = np.argmin(max_similarity_values)
        ref_structures.append(cand_structures[min_max_similarity_index])
        soap_ref = np.concatenate([soap_ref, soap_cand[min_max_similarity_index:min_max_similarity_index+1]], axis=0)
        soap_ref_self = np.concatenate((soap_ref_self, soap_cand_self[min_max_similarity_index][np.newaxis]))

        logger.info(f"Round {round_num}: Added structure with min max similarity {min_max_similarity:.5f}.")
        logger.info(f"Ref structures: {len(ref_structures)}, Cand structures: {len(cand_structures)}")
        logger.info("---------")

        # If reached the maximum rounds, stop the loop
        if max_fps_rounds is not None and round_num >= max_fps_rounds:
            logger.info("Maximum FPS rounds reached.")
            break

    logger.info("---------")

    return ref_structures, soap_ref


# Main function
def main(ref_file, cand_file, n_jobs, batch_size, r_cut, n_max, l_max, threshold, max_fps_rounds, save_soap, save_dir):
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(save_dir, formatted_time)

    total_logger = setup_total_logging(save_path)
    total_logger.info('Total Log begin')
    start_time = time.time()

    # Reading xyz files
    if ref_file == '':
        ref_structures = []
    else:
        ref_structures = read(ref_file, index=':')
    cand_structures = read(cand_file, index=':')

    # Grouping structures by chemical formula
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

    # Confirming species
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
        total_logger.info(f"Processing formula {i+1:>}/{formula_num:>}: {formula}")
        total_logger.info(f"Start Ref structures: {len(ref_dict[formula])}, Cand structures: {len(cand_dict[formula])}")

        logger = setup_logging(formula, save_path)
        formula_path = os.path.join(save_path, formula)
        formula_start_time = time.time()
        logger.info('Log begin')
        logger.info(f"Processing formula: {formula}")

        # Only use the chemical elements contained in the current chemical formula
        # species = list(set(cand_dict[formula][0].get_chemical_symbols()))
        updated_structures, updated_soap_list = compare_and_update_structures(ref_dict[formula], 
                                                                              cand_dict[formula], 
                                                                              n_jobs=n_jobs,
                                                                              batch_size=batch_size,                                                                                  
                                                                              species=species,
                                                                              r_cut=r_cut,
                                                                              n_max=n_max,
                                                                              l_max=l_max,
                                                                              threshold=threshold,
                                                                              max_fps_rounds=max_fps_rounds,
                                                                              save_soap=save_soap,
                                                                              logger=logger)
        
        # Saving the updated reference structures
        write(os.path.join(formula_path, f"updated_ref_structures_{formula}.xyz"), updated_structures)
        logger.info(f"Updated reference structures saved to '{formula}/updated_ref_structures_{formula}.xyz'")
        
        # Saving SOAP descriptors of the updated reference structures, if needed
        if save_soap:
            soap_dict = defaultdict(list)
            for i in range(len(updated_soap_list)):
                soap_result = updated_soap_list[i]
                soap_dict[formula].append(soap_result)
            save_soap_to_hdf5(soap_dict, os.path.join(formula_path, f"updated_ref_soap_descriptors_{formula}.h5"))
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
    parser = argparse.ArgumentParser(description='Select different chemical structures using farthest point sampling with SOAP samilarity.')
    parser.add_argument('--ref', type=str, default='', help='Reference XYZ file')
    parser.add_argument('--cand', type=str, required=True, help='Candidate XYZ file')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of jobs for CPU parallel processing. None for all cores')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for CPU parallel processing')
    parser.add_argument('--r_cut', type=float, default=10.0, help='Cutoff radius for soap descriptor')
    parser.add_argument('--n_max', type=int, default=6, help='Number of radial basis functions')
    parser.add_argument('--l_max', type=int, default=4, help='Maximum degree of spherical harmonics')
    parser.add_argument('--threshold', type=float, default=0.9, help='Similarity threshold')
    parser.add_argument('--max_fps_rounds', type=int, default=None, help='Maximum number of FPS rounds. None for unlimited')
    parser.add_argument('--save_soap', type=bool, default=False, help='Save SOAP descriptor or not. True or False')
    parser.add_argument('--save_dir', type=str, default='fps_results', help='Save directory')
    args = parser.parse_args()

    main(args.ref, args.cand, args.n_jobs, args.batch_size, args.r_cut, args.n_max, args.l_max, args.threshold, args.max_fps_rounds, args.save_soap, args.save_dir)
