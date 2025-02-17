# [Re] On the Real-World Feasibility of Node Injection Fairness Attacks

[//]: # (todo: add the link to our paper)

This repository includes our reproduction of the paper [***Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections***](https://arxiv.org/abs/2406.03052) as well as our extensions.

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

- Additionally you can find poisoned graphs for the attacks on the fairGNNs in the following google drive folder:
https://drive.google.com/drive/folders/1G6GrLr-sqKugrF9Q1oslv0j3rwhtnY0z?usp=sharing


## Run the code
Due to conflicting dependencies, our FairSIN experiments can be run from a separate repository:
https://github.com/sobek1886/RE-NIFA-FairSIN

Other than that, the Makefile contains targets to run each experiment in the original paper. Also, to run the experiments on the FairGNN models there are different run commands not contained by the Makefile. We will introduce them at the end of this section.
The output of each command will be stored in the `output` directory, and have the same name as the target.

For example, to run the main experiment of the paper, you can run the following command:

```
make nifa
```

To run all experiments in the paper, you can run the following command:

```
make all
```

To select the device to run the experiments on, you can set the variable device.
For example, to run the experiments on `cuda:0`, you can run:

```
make all DEVICE=0
```

If CUDA is not available, experiments will run on the CPU.

Below is an overview of the targets in the makefile, and their corresponding experiments:

- `nifa`: Run the main experiment of the paper (produces Table 2: _Attack performance of NIFA on different victim GNN models_).
- `defense`: Run the defense experiment of the paper (produces Figure 3: _Defense performance on Pokec-z with masking η training nodes with the highest uncertainty_).
- `hyperparameters`: Runs the targets below and produces a file combinging all results. Note that each individual target below also produces its own output file.
  - `hyperparameter_alpha`: Runs experiments on hyperparameter alpha (produces Figure A2).
  - `hyperparameter_beta`: Runs experiments on hyperparameter beta (produces Figure A3).
  - `hyperparameter_perturbation`: Runs experiments on node injection budget (produces Figure A4).
  - `hyperparameter_k`: Runs experiments on hyperparameter k: uncertainty threshold (produces Figure A5).
 
To run the experiments of the FairVGNN you need to (1) download the poisoned datasets from the google drive folder given (2) put the files into the location code/FairVGNN/dataset/reproducibility (3) run the folloing command (adjust dataset if needed) in the FairVGNN folder:
python fairvgnn.py --dataset='pokec_n_poisoned' --encoder='GCN' --clip_e=0.1 --d_epochs=5 --g_epochs=5 --c_epochs=5 --c_lr=0.01 --e_lr=0.001 --ratio=0.5 --epochs=200 --prop='spmm' --alpha=0.5 --clip_e=0.5 

To run the experiments of the FairGNN you need to (1) download the poisoned datasets from the google drive folder given (2) put the files into the location code/FairGNN/dataset/reproducibility (3) run the folloing command (adjust dataset and other hyperparameters if needed) in the src folder:
python train_fairGNN.py \
        --seed=42 \
        --epochs=2000 \
        --model=GAT \
        --sens_number=200 \
        --dataset='reproducibility/dblp_poisoned' \
        --num-hidden=128 \
        --acc=0.65 \
        --roc=0.7 \
        --alpha=4 \
        --beta=0.01 \
        --num-heads=1 \
        --reproducibility=1



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
