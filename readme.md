# Implementations for NIFA

[//]: # (todo: add the link to our paper)

This repository includes the implementations for our paper at NeurIPS 2024: [***Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections.***](https://arxiv.org/abs/2406.03052)

<img src="https://github.com/CGCL-codes/NIFA/blob/main/framework.png" alt="Framework of NIFA">

## Environments

[//]: # (todo: change name from fact.yml)
This project provides a conda environment for CUDA systems (`fact.yml`), as well as a 
CPU environment (`fact-cpu.yml`).
You can install the environment by running the following command:

```
conda env create -f fact.yml
```

## Datasets & Processed files

[//]: # (todo: data in zenodo)
- Due to size limitation, the processed datasets are stored in  [google drive](https://drive.google.com/file/d/1WJYj8K3_H3GmJg-RZeRsJ8Z64gt3qCnq/view?usp=drive_link) as `data.zip`. The datasets include Pokec-z, Pokec-n and DBLP. 

- Download and unzip the `data.zip`, and the full repository should be as follows:

  ```
  .
  ├── code
  │   ├── attack.py
  │   ├── main.py
  │   ├── model.py
  │   ├── run.sh
  │   └── utils.py
  ├── data
  │   ├── dblp.bin
  │   ├── pokec_n.bin
  │   └── pokec_z.bin
  ├── Makefile
  ├── readme.md
  ├── fact.yml
  └── fact_cpu.yml
  ```

## Run the code

The makefile contains targets to run each experiment in the original paper.
The output of each command will be stored in the `output` directory, and have the same name as the target.

For example, to run the main experiment of the paper, you can run the following command:

```
make nifa
```

To run all experiments in the paper, you can run the following command:

```
make all
```

Below is an overview of the targets in the makefile, and their corresponding experiments:

- `nifa`: Run the main experiment of the paper (produces Table 2: _Attack performance of NIFA on different victim GNN models_).
- `defense`: Run the defense experiment of the paper (produces Figure 3: _Defense performance on Pokec-z with masking η training nodes with the highest uncertainty_).
- `hyperparameters`: Runs the targets below and produces a file combinging all results. Note that each individual target below also produces its own output file.
  - `hyperparameter_alpha`: Runs experiments on hyperparameter alpha (produces Figure A2).
  - `hyperparameter_beta`: Runs experiments on hyperparameter beta (produces Figure A3).
  - `hyperparameter_perturbation`: Runs experiments on node injection budget (produces Figure A4).
  - `hyperparameter_k`: Runs experiments on hyperparameter k: uncertainty threshold (produces Figure A5).

## Licenses

[//]: # (todo: get a license)

This project is licensed under CC BY-NC-ND 4.0. To view a copy of this license, please visit http://creativecommons.org/licenses/by-nc-nd/4.0/

## BibTeX

If you like our work and use the model for your research, please cite our work as follows:

[//]: # (todo: add our paper)

```bibtex
@inproceedings{luo2024nifa,
author = {Luo, Zihan and Huang, Hong and Zhou, Yongkang and Zhang, Jiping and Chen, Nuo and Jin, Hai},
title = {Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections},
booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
year = {2024},
month = {October}
}
``` 
