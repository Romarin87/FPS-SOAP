Some simple scripts for operating chemical datasets

### 1. select_differect_chemical_structures.py 
- 脚本简介  
这是 Sixuan Mi 在 Bowen Li 的指导和代码思想帮助下完成的脚本（当然少不了 GPT 的功劳），它基于 [SOAP](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html#) 描述符 和 [AverageKernel](https://singroup.github.io/dscribe/latest/tutorials/similarity_analysis/kernels.html) 计算判断两个化学结构的相似性，并通过比较 候选XYZ (`--cand`) 与 参考XYZ (`--ref`) 中化学结构的相似性，将 候选XYZ 中的结构逐一加入到 参考XYZ 中，直到相似度低于阈值的点全部被加入 参考XYZ 中为止。这样我们就通过 候选XYZ 更新了 参考XYZ 数据集。

- 脚本参数  
&emsp;`--ref`&emsp;&emsp;str，参考XYZ数据路径，默认''，表示没有参考XYZ数据  
&emsp;`--cand`&emsp;&emsp;str，候选XYZ路径，必须填写  
&emsp;`--njobs`&emsp;&emsp;int，并行核数，默认 8  
&emsp;`--r_cut`&emsp;&emsp;float，SOAP 描述符截断半径参数，默认 10.0  
&emsp;`--n_max`&emsp;&emsp;int，SOAP 描述符基函数参数，默认 6  
&emsp;`--l_max`&emsp;&emsp;int，SOAP 描述符球谐函数参数，默认 4  
&emsp;`--threshold`&emsp;&emsp;float，相似度阈值，用户需要根据数据集本身的性质以及希望筛选掉的结构数量进行调整，默认 0.9  

- 简单测试  
1. 你可以在命令行中运行 `python select_differect_chemical_structures.py --cand test_dataset/rxn0000_all.xyz &` 这是最简单的输入，只写入必须的 `--cand` 参数，表示在默认参数下从 `test_dataset/rxn0000_all.xyz` 中选择出所有相互之间相似程度低于阈值的点，你可以将你的输出文件与 `test_result/1` 中的文件进行对比，二者应该是一致的。
2. 你可以在命令行中运行 `python select_differect_chemical_structures.py --ref test_dataset/rxn000x.xyz --cand test_dataset/rxn000x_all.xyz --njobs 16 --r_cut 5 --n_max 3 --l_max 3 --threshold 0.99 &` 这是最复杂的输入，所有参数都显式的给出，表示将 `--cand` 中的点与 `--ref` 中的点比较，逐一挑选最不相似的点加入 `--ref` 中，直到相似度低于阈值的点全部被加入 `--ref` 中。你可以将你的输出文件与 `test_result/2` 中的文件进行对比，二者应该是一致的。  
  
2024年10月30日
