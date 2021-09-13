# SCINet
[![Arxiv link](https://img.shields.io/badge/arXiv-Time%20Series%20is%20a%20Special%20Sequence%3A%20Forecasting%20with%20Sample%20Convolution%20and%20Interaction-%23B31B1B)](https://arxiv.org/pdf/2106.09305.pdf)
[![state-of-the-art](https://img.shields.io/badge/-STATE--OF--THE--ART-blue?logo=Accenture&labelColor=lightgrey)]()
![pytorch](https://img.shields.io/badge/-PyTorch-%23EE4C2C?logo=PyTorch&labelColor=lightgrey)


This is the original pytorch implementation for the following paper: [Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction](https://arxiv.org/pdf/2106.09305.pdf). If you find this repository useful for your work, please consider citing it as follows:

```
@article{liu2021SCINet,
  title={Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction},
  author={Liu, Minhao and Zeng, Ailing and Lai, Qiuxia and Xu, Qiang},
  journal={arXiv preprint arXiv:2106.09305},
  year={2021}
}
```

## Updates
[2021-09-14] SCINet is released! 

## Features
- [x] Support **11** popular time-series forecasting datasets.
- [x] Provide all pretrained models.
- [x] Provide all training logs.

## Dataset

We conduct the experiments on 11 popular time-series datasets, namely Electricity Transformer Temperature (ETTh1, ETTh2 and ETTm1) , Traffic, Solar-Energy, Electricity and Exchange Rate and PeMS (PEMS03, PEMS04, PEMS07 and PEMS08), ranging from power, energy, finance and traffic domains. 

### Overall information of the 11 datasets

| Datasets      | Variants | Timesteps | Granularity | Start time | Task Type   |
| ------------- | -------- | --------- | ----------- | ---------- | ----------- |
| ETTh1         | 7        | 17,420    | 1hour       | 7/1/2016   | Multi-step  |
| ETTh2         | 7        | 17,420    | 1hour       | 7/1/2016   | Multi-step  |
| ETTm1         | 7        | 69,680    | 15min       | 7/1/2016   | Multi-step  |
| PEMS03        | 358      | 26,209    | 5min        | 5/1/2012   | Multi-step  |
| PEMS04        | 307      | 16,992    | 5min        | 7/1/2017   | Multi-step  |
| PEMS07        | 883      | 28,224    | 5min        | 5/1/2017   | Multi-step  |
| PEMS08        | 170      | 17,856    | 5min        | 3/1/2012   | Multi-step  |
| Traffic       | 862      | 17,544    | 1hour       | 1/1/2015   | Single-step |
| Solar-Energy  | 137      | 52,560    | 1hour       | 1/1/2006   | Single-step |
| Electricity   | 321      | 26,304    | 1hour       | 1/1/2012   | Single-step |
| Exchange-Rate | 8        | 7,588     | 1hour       | 1/1/1990   | Single-step |


## Get start

### Requirements

Install the required package first:

```
cd SCINet
conda create -n scinet python=3.8
conda activate scinet
pip install -r requirements.txt
```

### Dataset preparation

All datasets can be downloaded [here](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing). To prepare all dataset at one time, you can just run:
```
source prepare_data.sh
```
[![ett](https://img.shields.io/badge/Download-ETT_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/1NU85EuopJNkptFroPtQVXMZE70zaBznZ)
[![pems](https://img.shields.io/badge/Download-PeMS_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/17fwxGyQ3Qb0TLOalI-Y9wfgTPuXSYgiI)
[![financial](https://img.shields.io/badge/Download-financial_Dataset-%234285F4?logo=GoogleDrive&labelColor=lightgrey)](https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-)

### Run training code

To facilitate reproduction, we provide the logs on the above datasets [here](https://drive.google.com/drive/folders/1MBK5MOShD4ygLIinNBo2F8EPRM5y9qIQ?usp=sharing) in details. You can check **the hyperparameters, training loss and test results for each epoch** in these logs as well.

We follow the same settings of [StemGNN](https://github.com/microsoft/StemGNN) for PEMS 03, 04, 07, 08 datasets, [MTGNN](https://github.com/nnzhan/MTGNN) for Solar, electricity, traffic, financial datasets, [Informer](https://github.com/zhouhaoyi/Informer2020) for ETTH1, ETTH2, ETTM1 datasets. The detailed training commands are given as follows.

#### For PEMS dataset (All datasets follow Input 12, Output 12):
```
# pems03
python run_pems.py --dataset PEMS03 --hidden-size 0.0625 --dropout 0.25 --model_name pems03_h0.0625_dp0.25

# pems04
python run_pems.py --dataset PEMS04 --hidden-size 0.0625 --dropout 0 --model_name pems04_h0.0625_dp0

# pems07
python run_pems.py --dataset PEMS07 --hidden-size 0.03125 --dropout 0.25 --model_name pems07_h0.03125_dp0.25

# pems08
python run_pems.py --dataset PEMS08 --hidden-size 1 --dropout 0.5 --model_name pems08_h1_dp0.5

```

##### PEMS Parameter highlights

| Parameter Name | Description             | Parameter in paper | Default |
| -------------- | ----------------------- | ------------------ | ------- |
| dataset        | Name of dataset      | N/A                | PEMS08  |
| horizon        | Horizon                 | Horizon            | 12      |
| window_size    | Look-back window        | Look-back window   | 12      |
| hidden-size    | hidden expansion        | h                  | 1       |
| layers         | SCINet block layers     | L                  | 3       |
| stacks         | The number of SCINet block| K                | 1       |


#### For Solar dataset:
```
# predict 3 
python run_financial.py --dataset_name solar_AL --window_size 160 --horizon 3 -hidden-size 2 --single_step 0 --lastWeight 0.5 --stacks 1 --layers 4 --num_concat 0 --lradj 6 --lr 1e-4 --dropout 0.25 --batch_size 1024 --model_name so_I160_o3_lr1e-4_bs1024_dp0.25_h2_s1l4_w0.5 --save_path so_I160_o3_lr1e-4_bs1024_dp0.25_h2_s1l4_w0.5

# predict 6
python run_financial.py --dataset_name solar_AL --window_size 160 --horizon 6 -hidden-size 2 --single_step 0 --lastWeight 0.5 --stacks 2 --layers 4 --num_concat 0 --lradj 6 --lr 1e-4 --dropout 0.25 --batch_size 1024 --model_name so_I160_o6_lr1e-4_bs1024_dp0.25_h2_s2l4_w0.5 --save_path so_I160_o6_lr1e-4_bs1024_dp0.25_h2_s2l4_w0.5

# predict 12
python run_financial.py --dataset_name solar_AL --window_size 160 --horizon 12 -hidden-size 2 --single_step 0 --lastWeight 0.5 --stacks 2 --layers 4 --num_concat 0 --lradj 6 --lr 1e-4 --dropout 0.25 --batch_size 1024 --model_name so_I160_o12_lr1e-4_bs1024_dp0.25_h2_s2l4_w0.5 --save_path so_I160_o12_lr1e-4_bs1024_dp0.25_h2_s2l4_w0.5

# predict 24
python run_financial.py --dataset_name solar_AL --window_size 160 --horizon 24 -hidden-size 2 --single_step 0 --lastWeight 0.5 --stacks 1 --layers 4 --num_concat 0 --lradj 6 --lr 1e-4 --dropout 0.25 --batch_size 1024 --model_name so_I160_o24_lr1e-4_bs1024_dp0.25_h2_s1l4_w0.5 --save_path so_I160_o24_lr1e-4_bs1024_dp0.25_h2_s1l4_w0.5
```

#### For Electricity dataset:

```
# predict 3 
python run_financial.py --dataset_name electricity --window_size 168 --horizon 3 --hidden-size 8 --single_step 1 --lastWeight 0.5 --stacks 2 --layers 3 --num_concat 0 --lradj 1 --lr 9e-3 --dropout 0 --batch_size 32 --model_name ele_I168_o3_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --save_path ele_I168_o3_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --groups 321

# predict 6
python run_financial.py --dataset_name electricity --window_size 168 --horizon 6 --hidden-size 8 --single_step 1 --lastWeight 0.5 --stacks 2 --layers 3 --num_concat 0 --lradj 1 --lr 9e-3 --dropout 0 --batch_size 32 --model_name ele_I168_o6_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --save_path ele_I168_o6_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --groups 321

# predict 12
python run_financial.py --dataset_name electricity --window_size 168 --horizon 12 --hidden-size 8 --single_step 1 --lastWeight 0.5 --stacks 2 --layers 3 --num_concat 0 --lradj 1 --lr 9e-3 --dropout 0 --batch_size 32 --model_name ele_I168_o12_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --save_path ele_I168_o12_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --groups 321

# predict 24
python run_financial.py --dataset_name electricity --window_size 168 --horizon 24 --hidden-size 8 --single_step 1 --lastWeight 0.5 --stacks 2 --layers 3 --num_concat 0 --lradj 1 --lr 9e-3 --dropout 0 --batch_size 32 --model_name ele_I168_o24_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --save_path ele_I168_o24_lr9e-3_bs32_dp0_h8_s2l3_w0.5 --groups 321
```

#### For Traffic dataset:

```
# predict 3 
python run_financial.py --dataset_name traffic --window_size 168 --horizon 3 --hidden-size 2 --single_step 1 --lastWeight 1.0 --stacks 2 --layers 3 --num_concat 0 --lradj 1 --lr 5e-4 --dropout 0.25 --batch_size 16 --model_name traf_I168_o3_lr5e-4_bs16_dp0.25_h2_s2l3_w1.0 --save_path traf_I168_o3_lr5e-4_bs16_dp0.25_h2_s2l3_w1.0

# predict 6
python run_financial.py --dataset_name traffic --window_size 168 --horizon 6 --hidden-size 2 --single_step 1 --lastWeight 1.0 --stacks 2 --layers 2 --num_concat 0 --lradj 1 --lr 5e-4 --dropout 0.25 --batch_size 16 --model_name traf_I168_o6_lr5e-4_bs16_dp0.25_h2_s2l2_w1.0 --save_path traf_I168_o6_lr5e-4_bs16_dp0.25_h2_s2l2_w1.0

# predict 12
python run_financial.py --dataset_name traffic --window_size 168 --horizon 12 --hidden-size 1 --single_step 1 --lastWeight 1.0 --stacks 2 --layers 3 --num_concat 0 --lradj 1 --lr 5e-4 --dropout 0.25 --batch_size 16 --model_name traf_I168_o12_lr5e-4_bs16_dp0.25_h1_s2l3_w1.0 --save_path traf_I168_o12_lr5e-4_bs16_dp0.25_h1_s2l3_w1.0

# predict 24
python run_financial.py --dataset_name traffic --window_size 168 --horizon 24 --hidden-size 2 --single_step 1 --lastWeight 1.0 --stacks 2 --layers 2 --num_concat 0 --lradj 1 --lr 5e-4 --dropout 0.5 --batch_size 16 --model_name traf_I168_o24_lr5e-4_bs16_dp0.5_h2_s2l2_w1.0 --save_path traf_I168_o24_lr5e-4_bs16_dp0.5_h2_s2l2_w1.0
```

#### For Exchange rate dataset:

```
# predict 3 
python run_financial.py --dataset_name exchange_rate --window_size 168 --horizon 3 --hidden-size 0.125 --single_step 0 --lastWeight 0.5 --stacks 1 --layers 3 --num_concat 0 --lradj 1 --lr 5e-3 --dropout 0.5 --batch_size 4 --model_name ex_I168_o3_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --save_path ex_I168_o3_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --epochs 150

# predict 6
python run_financial.py --dataset_name exchange_rate --window_size 168 --horizon 6 --hidden-size 0.125 --single_step 0 --lastWeight 0.5 --stacks 1 --layers 3 --num_concat 0 --lradj 1 --lr 5e-3 --dropout 0.5 --batch_size 4 --model_name ex_I168_o6_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --save_path ex_I168_o6_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --epochs 150

# predict 12
python run_financial.py --dataset_name exchange_rate --window_size 168 --horizon 12 --hidden-size 0.125 --single_step 0 --lastWeight 0.5 --stacks 1 --layers 3 --num_concat 0 --lradj 1 --lr 5e-3 --dropout 0.5 --batch_size 4 --model_name ex_I168_o12_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --save_path ex_I168_o12_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --epochs 150

# predict 24
python run_financial.py --dataset_name exchange_rate --window_size 168 --horizon 24 --hidden-size 0.125 --single_step 0 --lastWeight 0.5 --stacks 1 --layers 3 --num_concat 0 --lradj 1 --lr 5e-3 --dropout 0.5 --batch_size 4 --model_name ex_I168_o24_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --save_path ex_I168_o24_lr5e-3_bs4_dp0.5_h0.125_s1l3_w0.5 --epochs 150
```


##### Financial Parameter highlights

| Parameter Name | Description               | Parameter in paper      | Default                                |
| -------------- | ------------------------- | ----------------------- | -------------------------------------- |
| dataset_name           | Data name | N/A     | exchange_rate |
| horizon        | Horizon                   | Horizon                 | 3                                      |
| window_size    | Look-back window          | Look-back window        | 168                                    |
| batch_size     | Batch size                | batch size              | 8                                      |
| lr             | Learning rate             | learning rate           | 5e-3                                   |
| hidden-size    | hidden expansion          | h                       | 1                                      |
| layers         | SCINet block layers       | L                       | 3                                      |
| stacks         | The number of SCINet block| K                       | 1                                      |
| lastweight     | Loss weight of the last frame| Loss weight ($\lambda$) | 1.0                                 |


#### For ETTH1 dataset:

```
# multivariate, out 24
python run_ETTh.py --data ETTh1 --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --layers 3 --learning_rate 0.009 --batch_size 4 --dropout 0.5 --model_name etth1_M_I48_O24_lr0.009_bs4_dp0.5_h4_s1l3

# multivariate, out 48
python run_ETTh.py --data ETTh1 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 --layers 3 --learning_rate 0.009 --batch_size 16 --dropout 0.25 --model_name etth1_M_I96_O48_lr0.009_bs16_dp0.25_h4_s1l3

# multivariate, out 168
python run_ETTh.py --data ETTh1 --features M  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --layers 3 --learning_rate 5e-4 --batch_size 32 --dropout 0.5 --model_name etth1_M_I336_O168_lr5e-4_bs32_dp0.5_h4_s1l3

# multivariate, out 336
python run_ETTh.py --data ETTh1 --features M  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --layers 4 --learning_rate 1e-4 --batch_size 512 --dropout 0.5 --model_name etth1_M_I336_O336_lr1e-4_bs512_dp0.5_h1_s1l4

# multivariate, out 720
python run_ETTh.py --data ETTh1 --features M  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 1 --stacks 1 --layers 5 --learning_rate 1e-5 --batch_size 128 --dropout 0.5 --model_name etth1_M_I736_O720_lr1e-5_bs128_dp0.5_h1_s1l5

# Univariate, out 24
python run_ETTh.py --data ETTh1 --features S  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 8 --stacks 1 --layers 3 --learning_rate 0.007 --batch_size 16 --dropout 0.25 --model_name etth1_S_I48_O24_lr0.007_bs16_dp0.25_h8_s1l3

# Univariate, out 48
python run_ETTh.py --data ETTh1 --features S  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 --layers 4 --learning_rate 0.007 --batch_size 4 --dropout 0.5 --model_name etth1_S_I96_O48_lr0.007_bs4_dp0.5_h4_s1l4

# Univariate, out 168
python run_ETTh.py --data ETTh1 --features S  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 0.5 --stacks 1 --layers 4 --learning_rate 5e-5 --batch_size 8 --dropout 0.5 --model_name etth1_S_I336_O168_lr5e-5_bs8_dp0.5_h0.5_s1l4

# Univariate, out 336
python run_ETTh.py --data ETTh1 --features S  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --layers 4 --learning_rate 5e-5 --batch_size 128 --dropout 0.5 --model_name etth1_S_I336_O336_lr5e-5_bs128_dp0.5_h1_s1l4

# Univariate, out 720
python run_ETTh.py --data ETTh1 --features S  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 4 --stacks 1 --layers 5 --learning_rate 1e-5 --batch_size 32 --dropout 0.5 --model_name etth1_S_I736_O720_lr1e-5_bs32_dp0.5_h4_s1l5
```

#### For ETTH2 dataset:
```
# multivariate, out 24
python run_ETTh.py --data ETTh2 --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 8 --stacks 1 --layers 3 --learning_rate 0.007 --batch_size 16 --dropout 0.25 --model_name etth2_M_I48_O24_lr7e-3_bs16_dp0.25_h8_s1l3

# multivariate, out 48
python run_ETTh.py --data ETTh2 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 --layers 4 --learning_rate 0.007 --batch_size 4 --dropout 0.5 --model_name etth2_M_I96_O48_lr7e-3_bs4_dp0.5_h4_s1l4

# multivariate, out 168
python run_ETTh.py --data ETTh2 --features M  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 0.5 --stacks 1 --layers 4 --learning_rate 5e-5 --batch_size 8 --dropout 0.5 --model_name etth2_M_I336_O168_lr5e-5_bs8_dp0.5_h0.5_s1l4

# multivariate, out 336
python run_ETTh.py --data ETTh2 --features M  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 1 --stacks 1 --layers 4 --learning_rate 5e-5 --batch_size 128 --dropout 0.5 --model_name etth2_M_I336_O336_lr5e-5_bs128_dp0.5_h1_s1l4

# multivariate, out 720
python run_ETTh.py --data ETTh2 --features M  --seq_len 736 --label_len 720 --pred_len 720 --hidden-size 4 --stacks 1 --layers 5 --learning_rate 1e-5 --batch_size 32 --dropout 0.5 --model_name etth2_M_I736_O720_lr1e-5_bs32_dp0.5_h4_s1l5

# Univariate, out 24
python run_ETTh.py --data ETTh2 --features S  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --layers 3 --learning_rate 0.001 --batch_size 16 --dropout 0 --model_name etth2_S_I48_O24_lr1e-3_bs16_dp0_h4_s1l3

# Univariate, out 48
python run_ETTh.py --data ETTh2 --features S  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 2 --layers 4 --learning_rate 0.001 --batch_size 32 --dropout 0.5 --model_name etth2_S_I96_O48_lr1e-3_bs32_dp0.5_h4_s2l4

# Univariate, out 168
python run_ETTh.py --data ETTh2 --features S  --seq_len 336 --label_len 168 --pred_len 168 --hidden-size 4 --stacks 1 --layers 3 --learning_rate 1e-4 --batch_size 8 --dropout 0 --model_name etth2_S_I336_O168_lr1e-4_bs8_dp0_h4_s1l3

# Univariate, out 336
python run_ETTh.py --data ETTh2 --features S  --seq_len 336 --label_len 336 --pred_len 336 --hidden-size 8 --stacks 1 --layers 3 --learning_rate 5e-4 --batch_size 512 --dropout 0.5 --model_name etth2_S_I336_O336_lr5e-4_bs512_dp0.5_h8_s1l3

# Univariate, out 720
python run_ETTh.py --data ETTh2 --features S  --seq_len 720 --label_len 720 --pred_len 720 --hidden-size 8 --stacks 1 --layers 3 --learning_rate 5e-5 --batch_size 16 --dropout 0.6 --model_name etth2_S_I720_O720_lr5e-5_bs16_dp0.6_h8_s1l3
```

#### For ETTM1 dataset:
```
# multivariate, out 24
python run_ETTh.py --data ETTm1 --features M  --seq_len 48 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --layers 3 --learning_rate 0.005 --batch_size 32 --dropout 0.5 --model_name ettm1_M_I48_O24_lr7e-3_bs16_dp0.25_h8_s1l3

# multivariate, out 48
python run_ETTh.py --data ETTm1 --features M  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 2 --layers 4 --learning_rate 0.001 --batch_size 16 --dropout 0.5 --model_name ettm1_M_I96_O48_lr1e-3_bs16_dp0.5_h4_s2l4

# multivariate, out 96
python run_ETTh.py --data ETTm1 --features M  --seq_len 384 --label_len 96 --pred_len 96 --hidden-size 0.5 --stacks 2 --layers 4 --learning_rate 5e-5 --batch_size 32 --dropout 0.5 --model_name ettm1_M_I384_O96_lr5e-5_bs32_dp0.5_h0.5_s2l4

# multivariate, out 288
python run_ETTh.py --data ETTm1 --features M  --seq_len 672 --label_len 288 --pred_len 288 --hidden-size 4 --stacks 1 --layers 5 --learning_rate 1e-5 --batch_size 32 --dropout 0.5 --model_name ettm1_M_I672_O288_lr1e-5_bs32_dp0.5_h0.5_s1l5

# multivariate, out 672
python run_ETTh.py --data ETTm1 --features M  --seq_len 672 --label_len 672 --pred_len 672 --hidden-size 4 --stacks 2 --layers 5 --learning_rate 1e-5 --batch_size 32 --dropout 0.5 --model_name ettm1_M_I672_O672_lr1e-5_bs32_dp0.5_h4_s2l5

# Univariate, out 24
python run_ETTh.py --data ETTm1 --features S  --seq_len 96 --label_len 24 --pred_len 24 --hidden-size 4 --stacks 1 --layers 4 --learning_rate 0.001 --batch_size 8 --dropout 0 --model_name ettm1_S_I96_O24_lr1e-3_bs8_dp0_h4_s1l4

# Univariate, out 48
python run_ETTh.py --data ETTm1 --features S  --seq_len 96 --label_len 48 --pred_len 48 --hidden-size 4 --stacks 1 --layers 3 --learning_rate 0.005 --batch_size 16 --dropout 0 --model_name ettm1_S_I96_O48_lr5e-3_bs16_dp0_h4_s1l3

# Univariate, out 96
python run_ETTh.py --data ETTm1 --features S  --seq_len 384 --label_len 96 --pred_len 96 --hidden-size 2 --stacks 1 --layers 4 --learning_rate 1e-4 --batch_size 8 --dropout 0 --model_name ettm1_S_I384_O96_lr1e-4_bs8_dp0_h2_s1l4

# Univariate, out 288
python run_ETTh.py --data ETTm1 --features S  --seq_len 384 --label_len 288 --pred_len 288 --hidden-size 4 --stacks 1 --layers 4 --learning_rate 5e-5 --batch_size 64 --dropout 0 --model_name ettm1_S_I384_O288_lr5e-5_bs64_dp0_h4_s1l4

# Univariate, out 672
python run_ETTh.py --data ETTm1 --features S  --seq_len 672 --label_len 672 --pred_len 672 --hidden-size 1 --stacks 1 --layers 5 --learning_rate 1e-4 --batch_size 32 --model_name ettm1_S_I672_O672_lr1e-4_bs32_dp0.5_h1_s1l5
```


##### ETT Parameter highlights

| Parameter Name | Description                  | Parameter in paper | Default                    |
| -------------- | ---------------------------- | ------------------ | -------------------------- |
| root_path      | The root path of subdatasets | N/A                | './datasets/ETT-data/ETT/' |
| data           | Subdataset                   | N/A                | ETTh1                      |
| pred_len       | Horizon                      | Horizon            | 48                         |
| seq_len        | Look-back window             | Look-back window   | 96                         |
| batch_size     | Batch size                   | batch size         | 32                         |
| lr             | Learning rate                | learning rate      | 0.0001                     |
| hidden-size    | hidden expansion             | h                  | 1                          |
| layers         | SCINet block layers          | L                  | 3                          |
| stacks         | The number of SCINet blocks  | K                  | 1                          |

## Special Constrain

Because of the stacked binary down-sampling method that SCINet adapts, the max number of layers (L_max) is contrained by the look-back window (W) size as shown below:

$$L_{max}=\lfloor log_2 W \rfloor$$

## Contact

If you have any questions, feel free to contact us or post github issues. Pull requests are highly welcomed! 

```
Minhao Liu: mhliu@cse.cuhk.edu.hk
Ailing Zeng: alzeng@cse.cuhk.edu.hk
Zhijian Xu: zjxu21@cse.cuhk.edu.hk
```

## Acknowledgements
Thank you all for your attention to our work!

This code uses ([Informer](https://github.com/zhouhaoyi/Informer2020), [MTGNN](https://github.com/nnzhan/MTGNN), [StemGNN](https://github.com/microsoft/StemGNN)) as baseline methods for comparison. 
