# 让我们说中文
Some simple scripts for operating chemical datasets  

---

## 1. select_differect_chemical_structures.py  
- 脚本简介  
该脚本可以从一个数据集中根据几何坐标挑选出相似度低于给定阈值的所有结构，从而进行数据的精简。
此处脚本这是在原先并行 CPU 脚本上进行修改，目前支持 单GPU 运行与 单/多CPU 运行，在 GPU 运行速度明显快于原 24核CPU 的速度，单CPU 运行也远快于原脚本，但 多CPU 运行时可能会出现内存泄漏问题。

- 脚本参数  
&emsp;`--ref`&emsp;&emsp;str，参考XYZ数据路径，默认''，表示没有参考XYZ数据  
&emsp;`--cand`&emsp;&emsp;str，候选XYZ路径，必须填写  
&emsp;`--njobs`&emsp;&emsp;int，CPU并行核数，默认 1  
&emsp;`--gpu`&emsp;&emsp;int，使用的GPU数，默认 1（目前只能是 0 或 1）  
&emsp;`--batch_size`&emsp;&emsp;int，GPU运行时读取数据的批大小，默认 50  
&emsp;`--r_cut`&emsp;&emsp;float，SOAP 描述符截断半径参数，默认 10.0  
&emsp;`--n_max`&emsp;&emsp;int，SOAP 描述符基函数参数，默认 6  
&emsp;`--l_max`&emsp;&emsp;int，SOAP 描述符球谐函数参数，默认 4  
&emsp;`--threshold`&emsp;&emsp;float，相似度阈值，用户需要根据数据集本身的性质以及希望筛选掉的结构数量进行调整，默认 0.9  

---

## 2. select_differect_chemical_structures_cpu.py  
- 脚本简介  
原先并行 CPU 脚本，它基于 [SOAP](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html#) 描述符 和 [AverageKernel](https://singroup.github.io/dscribe/latest/tutorials/similarity_analysis/kernels.html) 计算判断两个化学结构的相似性，并通过比较 候选XYZ (`--cand`) 与 参考XYZ (`--ref`) 中化学结构的相似性，将 候选XYZ 中的结构逐一加入到 参考XYZ 中，直到相似度低于阈值的点全部被加入 参考XYZ 中为止。这样我们就通过 候选XYZ 更新了 参考XYZ 数据集。

- 脚本参数  
同上，但没有 &emsp;`--gpu`&emsp; 与 &emsp;`--batch_size`&emsp;，且 &emsp;`--njobs`&emsp; 默认为 8

- 简单测试  
1. 你可以在命令行中运行 `python select_differect_chemical_structures.py --cand test_dataset/rxn0000_all.xyz &` 这是最简单的输入，只写入必须的 `--cand` 参数，表示在默认参数下从 `test_dataset/rxn0000_all.xyz` 中选择出所有相互之间相似程度低于阈值的点，你可以将你的输出文件与 `test_result/1` 中的文件进行对比，二者应该是一致的。
2. 你可以在命令行中运行 `python select_differect_chemical_structures.py --ref test_dataset/rxn000x.xyz --cand test_dataset/rxn000x_all.xyz --njobs 16 --r_cut 5 --n_max 3 --l_max 3 --threshold 0.99 &` 这是最复杂的输入，所有参数都显式的给出，表示将 `--cand` 中的点与 `--ref` 中的点比较，逐一挑选最不相似的点加入 `--ref` 中，直到相似度低于阈值的点全部被加入 `--ref` 中。你可以将你的输出文件与 `test_result/2` 中的文件进行对比，二者应该是一致的。  

---
  
## 3. calculate_atomic_pair_distances.py  
- 脚本简介  
计算分子中两两原子对之间的距离。  

- 脚本参数  
&emsp;`--file`&emsp;&emsp;str，输入文件路径，必须填写，格式为 .xyz  
&emsp;`--ignore`&emsp;&emsp;str，忽略的元素列表，以逗号分隔（例如，H,O），默认是''  
&emsp;`--only`&emsp;&emsp;str，计算的元素对列表，以逗号分隔（例如，C-C,O-C），默认是''  
&emsp;`--output`&emsp;&emsp;str，输出 HDF5 文件名，必须填写  

---

## 4. draw_bond_distribution.py
- 脚本简介  
绘制分子中两两原子对之间的距离分布图。 

- 脚本参数  
&emsp;`--files`&emsp;&emsp;list，必须填写，一或多个键长分布文件名的列表  
&emsp;`--output`&emsp;&emsp;str，必须填写，输出图形的名称  
&emsp;`--range`&emsp;&emsp;str，默认 '1,5'，以逗号分隔的距离范围（单位：Å），例如 '1,5'  

---

## 5. calculate_coulomb_matrices.py  
- 脚本简介  
计算分子的库伦矩阵。  

- 脚本参数  
&emsp;`--file`&emsp;&emsp;str，必须填写，输入的 .xyz 格式文件  
&emsp;`--output`&emsp;&emsp;str，必须填写，输出的库仑矩阵文件名（HDF5 格式）  
&emsp;`--n_jobs`&emsp;&emsp;int，默认 1，计算时使用的并行作业数  

---

## 6. do_clustering.py   
- 脚本简介  
基于库伦矩阵，对分子进行降维聚类。  

- 脚本参数  
&emsp;`--files`&emsp;&emsp;list，必须填写，文件名列表  
&emsp;`--tsne`&emsp;&emsp;str，默认 None，自定义 t-SNE 参数（例如，"n_components=2,perplexity=30"）  
&emsp;`--umap`&emsp;&emsp;str，默认 None，自定义 UMAP 参数（例如，"n_components=2,n_neighbors=10"）  

---

## 7. draw_clustering_results.py  
- 脚本简介  
绘制分子的降维聚类图。   

- 脚本参数  
&emsp;`--tsne`&emsp;&emsp;str，与下一参数必须填写一个，t-SNE HDF5 文件路径  
&emsp;`--umap`&emsp;&emsp;str，与上一参数必须填写一个，UMAP HDF5 文件路径  
&emsp;`--labels`&emsp;&emsp;str，必须填写，标签 HDF5 文件路径  


2024年12月28日
