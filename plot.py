import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# first save the forecasted output on the testset in the inference stage
# then load them as follows:
gt = np.load('results/modelTCN_Levels4_hid32_dataETTh1_ftM_sl96_ll48_pl48_lr0.009_bs16_hid4.0_s1_l3_dp0.25_invFalse_itr0_ind0/true_scale.npy', allow_pickle=True)[:2810]
pred_tcn = np.load('results/TCN_Levels4_hid32_dataETTh1_ftM_sl96_ll48_pl48_lr0.009_bs16_hid4.0_s1_l3_dp0.25_invFalse_itr0_ind0/pred_scale.npy', allow_pickle=True)[:2810]
pred_informer = np.load('results/informer_ETTh1_ftM_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0/pred.npy', allow_pickle=True)[:2810]
pred_scinet = np.load('results/SCINET_ETTh1_ftM_sl96_ll48_pl48_mxTrue_test_lr0.009_bs16_hid4.0_s1_l3_dp0.25_invFalse_itr0/pred.npy', allow_pickle=True)[:2810]

index = 1388
for i in range(0,7):
    # plot 7 variates in ETTh1 dataset
    fig = plt.figure(figsize=(8,6))
    plt.title('The prediction results of Informer on ETTh1 with In 96 Out 48 Setting')
    plt.plot(pred_informer[index,:,i],color=(168/255, 218/255, 220/255), marker='v',label = 'Informer')
    plt.plot(pred_tcn[index,:,i],color=(69/255, 123/255, 157/255),marker='v', label = 'TCN')
    plt.plot(pred_scinet[index,:,i],color=(218/255, 85/255, 82/255), marker='v',label = 'SCINet')
    plt.plot(gt[index,:,i],color=(167/255, 30/255, 52/255), marker='o', label = 'Ground Truth')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc=9,fontsize=14,ncol=4)
    plt.grid(True)
    plt.xlabel('Future Time Steps', fontsize=14)
    plt.ylabel('Prediction Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('ETTh1_M_i96o48_denorm_i{}'.format(i),dpi=300)
    plt.close()
