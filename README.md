# FPS-SOAP: Farthest Point Sampling with SOAP Descriptors

FPS-SOAP is a set of scripts for efficient chemical dataset curation using **Farthest Point Sampling (FPS)** algorithm combined with **[Smooth Overlap of Atomic Positions (SOAP)](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html#)** descriptors. The tool helps identify structurally dissimilar compounds by calculating similarity scores between molecular geometries, enabling dataset pruning or expansion for machine learning applications in chemistry.


## 📄 Project Paper
[General reactive machine learning potentials for CHON elements](https://faculty.ecnu.edu.cn/_s34/zt2/main.psp) <!-- 请在此处添加项目相关论文链接 -->


## 🚀 Environment Setup
### Dependencies
- Python 3.10.18  
- Dscribe 2.1.1  
- ASE 3.25.0
- Numpy 2.2.6  (CPU version)
- PyTorch 2.7.1 (GPU version)  

### Installation
```bash
# Create environment from requirements.txt (only support CPU version)
conda create --name fps-soap --file requirements.txt

# Activate the environment
conda activate fps-soap

# Install PyTorch (if GPU version is required)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## 📜 Script Documentation

### 1. CPU Version: `fps_cpu_numpy.py`
#### Purpose
Optimized CPU implementation for FPS-based structure similarity sampling using NumPy. Uses **[Laplacian kernel](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html) (accelerated by Numba JIT)** for atomic similarity calculation and **[AverageKernel](https://singroup.github.io/dscribe/latest/doc/dscribe.kernels.html#dscribe.kernels.averagekernel.AverageKernel)** for molecular similarity aggregation.

#### Key Parameters
| Parameter           | Type         | Default       | Description                                                                 |
|---------------------|--------------|---------------|-----------------------------------------------------------------------------|
| `--ref`             | str          | `""`          | Path to reference XYZ file (optional)                                      |
| `--cand`            | str          | required      | Path to candidate XYZ file (must be provided)                             |
| `--n_jobs`          | int          | `None`        | Number of CPU cores for parallel processing (`None` = all available cores) |
| `--batch_size`      | int          | 50            | Batch size for CPU parallel processing                                    |
| `--r_cut`           | float        | 10.0          | Cutoff radius for SOAP descriptor (unit: Å)                               |
| `--n_max`           | int          | 6             | Number of radial basis functions for SOAP descriptor                      |
| `--l_max`           | int          | 4             | Maximum degree of spherical harmonics for SOAP descriptor                 |
| `--threshold`       | float        | 0.9           | Similarity threshold (0-1, structures above this threshold are masked)  |
| `--dynamic_species` | bool         | `False`       | Use only chemical elements in the current formula (enable with `--dynamic_species`) |
| `--max_fps_rounds`  | int          | `None`        | Maximum number of FPS rounds (`None` = unlimited)                         |
| `--save_soap`       | bool         | `False`       | Save calculated SOAP descriptors to .h5 file (enable with `--save_soap`)   |
| `--save_dir`        | str          | `fps_results` | Directory to save output results (default: creates `fps_results/[timestamp]/formula` folders) |

#### Features
<!-- 此处需要完善 -->
- Automatically initializes reference set with first candidate structure if `--ref` is empty
- Parallelizes across CPU cores for similarity calculation using Numba
- Efficiently handles different sizes of datasets by adjusting `--threshold` and `--max_fps_rounds`



### 2. GPU Version: `fps_gpu_torch.py` (in development)
#### Purpose
GPU-accelerated version using PyTorch for FPS-based structure similarity sampling.  
**NOTE: GPU version now is significantly SLOWER than CPU version, not recommended!**

<!-- 
#### Key Parameters  
| Parameter       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `--gpu`         | GPU device index (0 or 1, default: 0)                                      |
| `--batch_size`  | Batch size for GPU inference (default: 50)                                 |
| `--njobs`       | CPU cores for preprocessing (default: 1)                                   |
| *Other params*  | Same as CPU version (see above)                                            |

#### Usage Example
```bash
python fps_gpu_torch.py \
  --cand large_dataset.xyz \
  --gpu 0 \
  --batch_size 100 \
  --threshold 0.9
```

#### Notes
- Single-GPU only support (multi-GPU coming soon)
- Avoid using `--njobs > 1` to prevent memory leaks
- For CPU-only run, set `--gpu -1` -->


## 🧪 Testing
### Test Command (CPU)
```bash
python fps_cpu_numpy.py \
  --cand tests/test_dataset/rxn000x_all.xyz \
  --save_dir tests/test_result/ \
  --threshold 0.99 
```

**Expected Output**:  
- Matching output files in `tests/test_result/` (compare with baseline)
- 8-core CPU runtime: ~10 seconds

<!-- ### Test Command (GPU)
<<<bash
python fps_gpu_torch.py \
  --cand tests/test_dataset/rxn000x_all.xyz \
  --gpu 0 \
  --threshold 0.99
<<<

**Expected Output**:  
- Same structure selection as CPU version
- GPU runtime: ~3-5 seconds (NVIDIA RTX 3090) -->


## 📁 Result Structure
Default outputs are saved in:  
```
fps_results/
└── YYYY-MM-DD-HH-MM-SS/
    └── total_output.log                              # Total Log file
    └── Formula/
        ├── updated_ref_structures_Formula.xyz        # Filtered XYZ file
        ├── updated_ref_soap_descriptors_Formula.h5   # (Optional) Saved SOAP descriptors
        └── Formula_output.log                        # Log file
```


## 📝 Citation
If you use this tool in your research, please cite:  
```bibtex
@article{BowenLi-2025,
  title={General reactive machine learning potentials for CHON elements},
  author={Bowen Li, Sixuan Mi, Jin Xiao, Shuwen Zhang, Han Wang, Tong Zhu},
  journal={Nature Computational Science (Ready to submit)},
  year={2025},
  doi={10.XXXX/XXXX}
}
```

---

**Last updated**: 2025-06-06