# SCINet
[![Arxiv link](https://img.shields.io/badge/arXiv-Time%20Series%20is%20a%20Special%20Sequence%3A%20Forecasting%20with%20Sample%20Convolution%20and%20Interaction-%23B31B1B)](https://arxiv.org/pdf/2106.09305.pdf)
[![state-of-the-art](https://img.shields.io/badge/-STATE--OF--THE--ART-blue?logo=Accenture&labelColor=lightgrey)]()
![pytorch](https://img.shields.io/badge/-PyTorch-%23EE4C2C?logo=PyTorch&labelColor=lightgrey)

## Some preface
This is the original pytorch implementation in the following paper: [Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction](https://arxiv.org/pdf/2106.09305.pdf). 

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

### Rank in Paper with Code

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/univariate-time-series-forecasting-on)](https://paperswithcode.com/sota/univariate-time-series-forecasting-on?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth1-168)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-168?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth1-24)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-24?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth1-336)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-336?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth1-48)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-48?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth1-720)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-720?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth2-168)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-168?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth2-24)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-24?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth2-336)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-336?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth2-48)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-48?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-etth2-720)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-720?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/traffic-prediction-on-pems04)](https://paperswithcode.com/sota/traffic-prediction-on-pems04?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-pemsd4)](https://paperswithcode.com/sota/time-series-forecasting-on-pemsd4?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-pemsd7)](https://paperswithcode.com/sota/time-series-forecasting-on-pemsd7?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/time-series-forecasting-on-pemsd8)](https://paperswithcode.com/sota/time-series-forecasting-on-pemsd8?p=time-series-is-a-special-sequence-forecasting)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/time-series-is-a-special-sequence-forecasting/univariate-time-series-forecasting-on-solar)](https://paperswithcode.com/sota/univariate-time-series-forecasting-on-solar?p=time-series-is-a-special-sequence-forecasting)


The following ranking data are retrieved from the [Paper with Code](https://paperswithcode.com/paper/time-series-is-a-special-sequence-forecasting) page on 2021 Sep 11th.  


## Get start

### Requirements

Install the required package first:

```
cd SCINET
conda create -n scinet python=3.8
conda activate scinet
pip install -r requirements.txt
```

### Dataset preparation

All datasets can be downloaded [here](https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2?usp=sharing). You can put the data you need into  the **dataset/** file.


### Run training code

To facility the reproduction, we provide the logs on the above datasets [here](https://drive.google.com/drive/folders/1MBK5MOShD4ygLIinNBo2F8EPRM5y9qIQ?usp=sharing) in details. You can check **the hyperparameters, training loss and test results for each epoch** in these logs as well.

We follow the same settings of [StemGNN](https://github.com/microsoft/StemGNN) for PEMS 03, 04, 07, 08 datasets, [MTGNN](https://github.com/nnzhan/MTGNN) for Solar, electricity, traffic, financial datasets, [Informer]((https://github.com/zhouhaoyi/Informer2020) for ETTH1, ETTH2, ETTM1 datasets. The detailed training commands are given as follows.

For PEMS dataset:
```
# pems03


# pems04

# pems07

# pems08

```

### PEMS Parameter highlights

| Parameter Name | Description             | Parameter in paper | Default |
| -------------- | ----------------------- | ------------------ | ------- |
| dataset        | Name of subdataset      | N/A                | PEMS08  |
| horizon        | Horizon                 | Horizon            | 12      |
| window_size    | Look-back window        | Look-back window   | 12      |
| batch_size     | Batch size              | batch size         | 8       |
| lr             | Learning rate           | learning rate      | 0.001   |
| hidden-size    | hidden expansion        | h                  | 1       |
| kernel         | convolution kernel size | k                  | 5       |
| layers         | SCINet block layers     | L                  | 3       |
| stacks         |                         | K                  | 1       |


For Solar dataset:
```

```

For Electricity dataset:

```

```

For Traffic dataset:

```

```

For Exchange rate dataset:

```

```


### Financial Parameter highlights

| Parameter Name | Description               | Parameter in paper      | Default                                |
| -------------- | ------------------------- | ----------------------- | -------------------------------------- |
| data           | loaction of the data file | N/A                     | ./datasets/financial/exchange_rate.txt |
| horizon        | Horizon                   | Horizon                 | 3                                      |
| window_size    | Look-back window          | Look-back window        | 168                                    |
| batch_size     | Batch size                | batch size              | 8                                      |
| lr             | Learning rate             | learning rate           | 5e-3                                   |
| hidden-size    | hidden expansion          | h                       | 1                                      |
| kernel         | convolution kernel size   | k                       | 5                                      |
| layers         | SCINet block layers       | L                       | 3                                      |
| stacks         | The number of SCINet block| K                       | 1                                      |
| lastweight     | Loss weight of the last frame| Loss weight ($\lambda$) | 1.0                                 |


For ETTH1 dataset:

```
# multivariate, out 24
python run_ETTh.py --data ETTh1 --features S  --hidden-size 4 --layers 3 --stacks 1 --seq_len 48 --label_len 24 --pred_len 24 --num_concat 0 --learning_rate 0.005 --kernel 5 --batch_size 32 --dropout 0.5 --model_name etth1_I48_out24_type1_lr0.005_bs32_dp0.5_h4_s1l3_e100 

# multivariate, out 48

# multivariate, out 168

# multivariate, out 336

# multivariate, out 720

# Univariate, out 24

# Univariate, out 48

# Univariate, out 168

# Univariate, out 336

# Univariate, out 720

```
For ETTH2 dataset:
```
python run_ETTh.py --data ETTh2 --features S  --hidden-size 4 --layers 3 --stacks 1 --seq_len 48 --label_len 24 --pred_len 24 --num_concat 0 --learning_rate 0.005 --kernel 5 --batch_size 32 --dropout 0.5 --model_name etth2_I48_out24_type1_lr0.005_bs32_dp0.5_h4_s1l3_e100

# multivariate, out 48

# multivariate, out 168

# multivariate, out 336

# multivariate, out 720

# Univariate, out 24

# Univariate, out 48

# Univariate, out 168

# Univariate, out 336

# Univariate, out 720
```

For ETTM1 dataset:
```
python run_ETTh.py --data ETTm1 --features S  --hidden-size 4 --layers 3 --stacks 1 --seq_len 48 --label_len 24 --pred_len 24 --num_concat 0 --learning_rate 0.005 --kernel 5 --batch_size 32 --dropout 0.5 --model_name etth2_I48_out24_type1_lr0.005_bs32_dp0.5_h4_s1l3_e100

# multivariate, out 48

# multivariate, out 168

# multivariate, out 336

# multivariate, out 720

# Univariate, out 24

# Univariate, out 48

# Univariate, out 168

# Univariate, out 336

# Univariate, out 720

```


### ETT Parameter highlights

| Parameter Name | Description                  | Parameter in paper | Default                    |
| -------------- | ---------------------------- | ------------------ | -------------------------- |
| root_path      | The root path of subdatasets | N/A                | './datasets/ETT-data/ETT/' |
| data           | Subdataset                   | N/A                | ETTh1                      |
| data_path      | Location of the data file    | N/A                | 'ETTh1.csv'                |
| pred_len       | Horizon                      | Horizon            | 48                         |
| seq_len        | Look-back window             | Look-back window   | 96                         |
| batch_size     | Batch size                   | batch size         | 32                         |
| lr             | Learning rate                | learning rate      | 0.0001                     |
| hidden-size    | hidden expansion             | h                  | 1                          |
| kernel         | convolution kernel size      | k                  | 5                          |
| layers         | SCINet block layers          | L                  | 3                          |
| stacks         | The number of SCINet blocks  | K                  | 1                          |







## Citation 

If you find this repository useful in your research, please consider citing the following paper:

```
@article{liu2021time,
  title={Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction},
  author={Liu, Minhao and Zeng, Ailing and Lai, Qiuxia and Xu, Qiang},
  journal={arXiv preprint arXiv:2106.09305},
  year={2021}
}
```

## Contact

If you have any questions, feel free to contact us or release github issues. Pull requests are highly welcomed! 

```
Minhao Liu: mhliu@cse.cuhk.edu.hk
Ailing Zeng: alzeng@cse.cuhk.edu.hk
Zhijian Xu: zjxu21@cse.cuhk.edu.hk
```

## Acknowledgements
This code uses ([Informer](https://github.com/zhouhaoyi/Informer2020), [MTGNN](https://github.com/nnzhan/MTGNN), [StemGNN](https://github.com/microsoft/StemGNN) as our baseline. We gratefully appreciate the impact these libraries had on this work. If you use our code, please consider citing the original papers as well. 

We also welcome all contributions to improve this library as well as give valuable feedbacks. We wish that this toolbox could serve the growing research community to reimplement existing methods and develop their own new models.
