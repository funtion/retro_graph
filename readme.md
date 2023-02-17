# Environment setup
Our code is modified from the [Retro*](https://github.com/binghong-ml/retro_star) and [SIRP](https://github.com/junsu-kim97/self_improved_retro). To set up enviroment, you need:
```bash
# Create environment
conda env create -n retrograph python=3.7 && conda activate retrograph

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit==11.3.1 cudatoolkit-dev==11.3.1 -c pytorch -c conda-forge -y
conda install rdkit -c rdkit -y
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install pytorch-lightning setuptools==59.5.0

# Install
pip install -e code/retro_star/packages/mlp_retrosyn
pip install -e code/retro_star/packages/rdchiral
pip install -e code
```



# Data and checkpoint
The original USPTO data can be downloaded from the [Retro*](https://github.com/binghong-ml/retro_star).
To ease of downloading, we also provide a copy in our [Zendo Repo](https://zenodo.org/record/7648612).
Meanwhile, you can also find our `USPTO-EXT` data and trained checkpoints in Zendo.

The `data` folder should have the following files
```bash
data
├── gnn.ckpt
├── one_step.ckpt
├── origin_dict.csv
├── routes_possible_test_hard.pkl
├── template_rules_1.dat
└── uspto_ext.pkl
```

Plase put the downloaded data under the `data` folder
# Search
To search on the USPTO dataset, run command

```bash
python code/retro_star/alg/retro_graph.py \
--mlp_model_dump data/one_step.ckpt \
--test_routes data/routes_possible_test_hard.pkl \
--starting_molecules data/origin_dict.csv \
--mlp_templates data/template_rules_1.dat \
--max_succes_count 1 --n_proc 12 \
--cluster_method 'none' --search_bsz 1 \
--use_gnn_plan --gnn_ckpt data/gnn.ckpt \
--gnn_dim 128 --gnn_dropout 0.1 --gnn_layers 3 --gnn_ratio 1.0
```

The expected output is
```bash

{10: 58, 20: 80, 30: 109, 40: 124, 50: 132, 100: 168, 200: 186, 300: 188, 400: 189, 500: 189}
{10: 0.30526315789473685, 20: 0.42105263157894735, 30: 0.5736842105263158, 40: 0.6526315789473685, 50: 0.6947368421052632, 100: 0.8842105263157894, 200: 0.9789473684210527, 300: 0.9894736842105263, 400: 0.9947368421052631, 500: 0.9947368421052631}
Iter 8574 / 190 = 45.126315789473686
```


# GNN training
Please refer `code/retro_star/alg/train_policy_gnn.py` and `code/retro_star/create_gnn_policy_data.py` for more details.

# Cite
If you find this work useful, we sincely appreciate if you can cite our KDD paper.
```bibtext
@inproceedings{xie2022retrograph,
  title={Retrograph: Retrosynthetic planning with graph search},
  author={Xie, Shufang and Yan, Rui and Han, Peng and Xia, Yingce and Wu, Lijun and Guo, Chenjuan and Yang, Bin and Qin, Tao},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2120--2129},
  year={2022}
}
```
