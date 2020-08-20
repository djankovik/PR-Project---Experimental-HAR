import re
import pandas as pd
import numpy as np
import zipfile
import os
import math
import statistics
import scipy.stats as stats
from scipy.stats import kurtosis,skew
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils import *
from statsmodels.tsa.ar_model import AutoReg
import sys
import pywt
import matplotlib.pyplot as plt

LABEL_NAMES = ['Sitting on bed', 'Sitting on table', 'Walking out', 'Walking in', 'Walking around', 'Eating at table', 'Laying on bed', 'No activity', 'Standing still']
SIGNAL_NAMES = ['MR left','PIR left','MR right','PIR right','MR up','PIR up','MR vector length','PIR vector length','MR alpha','MR beta','MR gama','PIR alpha','PIR beta','PIR gama']

np.seterr(divide='ignore', invalid='ignore')
headers = 'mean_1,std_1,max_1,min_1,iqr_1,entrophy_1,energy_1,fft_max_1,fft_min_1,fft_mean_1,maxInds_1,kurtosis_1,skewness_1,arcoef1_1,arcoef2_1,arcoef3_1,arcoef4_1,mean_2,std_2,max_2,min_2,iqr_2,entrophy_2,energy_2,fft_max_2,fft_min_2,fft_mean_2,maxInds_2,kurtosis_2,skewness_2,arcoef1_2,arcoef2_2,arcoef3_2,arcoef4_2,mean_3,std_3,max_3,min_3,iqr_3,entrophy_3,energy_3,fft_max_3,fft_min_3,fft_mean_3,maxInds_3,kurtosis_3,skewness_3,arcoef1_3,arcoef2_3,arcoef3_3,arcoef4_3,mean_4,std_4,max_4,min_4,iqr_4,entrophy_4,energy_4,fft_max_4,fft_min_4,fft_mean_4,maxInds_4,kurtosis_4,skewness_4,arcoef1_4,arcoef2_4,arcoef3_4,arcoef4_4,mean_5,std_5,max_5,min_5,iqr_5,entrophy_5,energy_5,fft_max_5,fft_min_5,fft_mean_5,maxInds_5,kurtosis_5,skewness_5,arcoef1_5,arcoef2_5,arcoef3_5,arcoef4_5,mean_6,std_6,max_6,min_6,iqr_6,entrophy_6,energy_6,fft_max_6,fft_min_6,fft_mean_6,maxInds_6,kurtosis_6,skewness_6,arcoef1_6,arcoef2_6,arcoef3_6,arcoef4_6,mean_7,std_7,max_7,min_7,iqr_7,entrophy_7,energy_7,fft_max_7,fft_min_7,fft_mean_7,maxInds_7,kurtosis_7,skewness_7,arcoef1_7,arcoef2_7,arcoef3_7,arcoef4_7,mean_8,std_8,max_8,min_8,iqr_8,entrophy_8,energy_8,fft_max_8,fft_min_8,fft_mean_8,maxInds_8,kurtosis_8,skewness_8,arcoef1_8,arcoef2_8,arcoef3_8,arcoef4_8,mean_9,std_9,max_9,min_9,iqr_9,entrophy_9,energy_9,fft_max_9,fft_min_9,fft_mean_9,maxInds_9,kurtosis_9,skewness_9,arcoef1_9,arcoef2_9,arcoef3_9,arcoef4_9,mean_10,std_10,max_10,min_10,iqr_10,entrophy_10,energy_10,fft_max_10,fft_min_10,fft_mean_10,maxInds_10,kurtosis_10,skewness_10,arcoef1_10,arcoef2_10,arcoef3_10,arcoef4_10,mean_11,std_11,max_11,min_11,iqr_11,entrophy_11,energy_11,fft_max_11,fft_min_11,fft_mean_11,maxInds_11,kurtosis_11,skewness_11,arcoef1_11,arcoef2_11,arcoef3_11,arcoef4_11,mean_12,std_12,max_12,min_12,iqr_12,entrophy_12,energy_12,fft_max_12,fft_min_12,fft_mean_12,maxInds_12,kurtosis_12,skewness_12,arcoef1_12,arcoef2_12,arcoef3_12,arcoef4_12,mean_13,std_13,max_13,min_13,iqr_13,entrophy_13,energy_13,fft_max_13,fft_min_13,fft_mean_13,maxInds_13,kurtosis_13,skewness_13,arcoef1_13,arcoef2_13,arcoef3_13,arcoef4_13,mean_14,std_14,max_14,min_14,iqr_14,entrophy_14,energy_14,fft_max_14,fft_min_14,fft_mean_14,maxInds_14,kurtosis_14,skewness_14,arcoef1_14,arcoef2_14,arcoef3_14,arcoef4_14,mad_mean_micro,mad_std_micro,mad_max_micro,mad_min_micro,mad_mean_pir,mad_std_pir,mad_max_pir,mad_min_pir,sma_micro,sma_pir,corr_m_xy,corr_m_yz,corr_m_xz,corr_p_xy,corr_p_yz,corr_p_xz,corr_mp_xx,corr_mp_yy,corr_mp_zz,class'
# print(len(headers.split(',')))

def sma(x,y,z):
    sum = 0
    for i in range(0,len(x)):
        sum += abs(x[i]) + abs(y[i]) + abs(z[i])
    return sum / len(x)

def get_features_for_windows(windows):
    res = []
    for window in windows:
        res.append(extract_features_from_window(window))
    return res

def write_sensor_measures_to_file(windows,fname):
    fnames={0:'microR_L',1:'pir_L',2:'microR_R',3:'pir_R',4:'microR_U',5:'pir_U',6:'microR_vector_len',7:'pir_vector_len',8:'microR_alpha',9:'microR_beta',10:'microR_gamma',11:'pir_alpha',12:'pir_beta',13:'pir_gama'}
    for i in range(0,len(windows[0][0])):
        f = open("wavelet_data\\"+fname+"\\init\\"+fnames[i]+".txt", 'w')
        sys.stdout = f
        for window in windows:
            column = [item[i] for item in window]
            line = ''
            for c in column:
                if c >= 0:
                    line+=' '
                line+=' '+str(c)
            #print(line)
        f.close()

def write_in_out_weka_style(input,output,fname):
    print('write_features_to_file')
    features = get_features_for_windows(input)
    f = open("wavelet_data\\"+fname+".csv", 'w')
    sys.stdout = f
    print(headers)
    for (fw,a) in zip(features,output):
        fw_s=''+str(fw[0])
        for x in fw[1:]:
            fw_s+=','+str(x)
        fw_s+=','+str(activies[a-1])
        print(fw_s)
    f.close()

def write_features_to_file(input,output,fname):
    # print('write_features_to_file')
    features = get_features_for_windows(input)
    f = open("wavelet_data\\"+fname+"\\X_"+fname+".txt", 'w')
    sys.stdout = f
    for fw in features:
        fw_s=''
        for x in fw:
            if x >= 0:
                fw_s+=' '
            fw_s+=' '+str(x)
        print(fw_s)
    f.close()
    f = open("wavelet_data\\"+fname+"\\y_"+fname+".txt", 'w')
    sys.stdout = f
    for a in output:
        print(a)
    f.close()

def extract_features_from_window(window_measurements):
    features = []
    for i in range(0,len(window_measurements[0])):
        column = [item[i] for item in window_measurements]
        mean = np.mean(np.array(column).astype(np.float))
        std = np.std(np.array(column).astype(np.float))
        max = np.amax(column)
        min = np.amin(column)
        
        pks = [i*1.0/np.sum(column) for i in column]
        entrophy = -np.sum([e*np.log2(abs(e)) for e in pks])
        
        iqrange = stats.iqr(column,axis=0)
        model = AutoReg(column,lags=int(len(column)/3))
        model_fit = model.fit()
        arrcoefs = model_fit.params[0:4]
        energy=np.sum([x*x for x in column])*1.0/len(column)

        fft = np.fft.fft(column)
        magn = [np.abs(c) for c in fft]
        fft_max=np.amax(np.array(magn).astype(np.float))
        fft_min=np.amin(np.array(magn).astype(np.float))
        fft_mean=np.mean(np.array(magn).astype(np.float))
        maxInds = np.argmax(np.array(magn).astype(np.float))

        kurt = kurtosis(column)
        skewn = skew(column)
        kur=np.mean(kurt)
        skwn = np.mean(skewn)

        features.extend([mean,std,max,min,iqrange,entrophy,energy,fft_max,fft_min,fft_mean,maxInds,kur,skwn])
        features.extend(arrcoefs)
    
    mad_m = stats.median_absolute_deviation(np.array([[item[0] for item in window_measurements],[item[2] for item in window_measurements],[item[4] for item in window_measurements]]))
    mad_p = stats.median_absolute_deviation(np.array([[item[1] for item in window_measurements],[item[3] for item in window_measurements],[item[5] for item in window_measurements]]))
    sma_m = sma([item[0] for item in window_measurements],[item[2] for item in window_measurements],[item[4] for item in window_measurements])
    sma_p = sma([item[1] for item in window_measurements],[item[3] for item in window_measurements],[item[5] for item in window_measurements])
    correlation_m_xy =np.corrcoef([item[0] for item in window_measurements],[item[2] for item in window_measurements])
    correlation_m_yz =np.corrcoef([item[2] for item in window_measurements],[item[4] for item in window_measurements])
    correlation_m_xz =np.corrcoef([item[0] for item in window_measurements],[item[4] for item in window_measurements])
    correlation_p_xy =np.corrcoef([item[0] for item in window_measurements],[item[2] for item in window_measurements])
    correlation_p_yz =np.corrcoef([item[2] for item in window_measurements],[item[4] for item in window_measurements])
    correlation_p_xz = np.corrcoef([item[0] for item in window_measurements],[item[4] for item in window_measurements])

    correlation_mp_xx =np.corrcoef([item[0] for item in window_measurements],[item[1] for item in window_measurements])
    correlation_mp_yy =np.corrcoef([item[2] for item in window_measurements],[item[3] for item in window_measurements])
    correlation_mp_zz =np.corrcoef([item[4] for item in window_measurements],[item[5] for item in window_measurements])

    correl_mp_xx = correlation_mp_xx[0,1]
    if str(correlation_mp_xx[0,1]) == 'nan':
        correl_mp_xx = 0.0
        
    features.extend([np.mean(mad_m),np.std(mad_m),np.amax(mad_m),np.amin(mad_m)])
    features.extend([np.mean(mad_p),np.std(mad_p),np.amax(mad_p),np.amin(mad_p)])
    features.extend([sma_m,sma_p])
    features.extend([correlation_m_xy[0,1],correlation_m_yz[0,1],correlation_m_xz[0,1],correlation_p_xy[0,1],correlation_p_yz[0,1],correlation_p_xz[0,1],correl_mp_xx,correlation_mp_yy[0,1],correlation_mp_zz[0,1]])

    return features

def scale_train_test_sets(X_train,X_test):
    history = {}
    for i in range(0,len(X_train[0][0])):
        column = []
        for window in X_train:
            column.extend([item[i] for item in window])
        measurements_mean = np.mean(np.array(column).astype(np.float))
        measurements_std = np.std(np.array(column).astype(np.float))
        history[i]={'mean':measurements_mean,'std':measurements_std}

    scaled_train = []
    for window in X_train:
        scaled_window = []
        for moment in window:
            scaled_moment=[]
            for i in range(0,len(X_train[0][0])):
                scaled_value = ((float(moment[i])-float(history[i]['mean']))/float(history[i]['std']))
                scaled_moment.append(scaled_value)
            scaled_window.append(scaled_moment)
        scaled_train.append(scaled_window)
    
    scaled_test = []
    for window in X_test:
        scaled_window = []
        for moment in window:
            scaled_moment=[]
            for i in range(0,len(X_train[0][0])):
                scaled_value = ((float(moment[i])-float(history[i]['mean']))/float(history[i]['std']))
                scaled_moment.append(scaled_value)
            scaled_window.append(scaled_moment)
        scaled_test.append(scaled_window)
    
    return (scaled_train,scaled_test)
    

def most_frequent(lst): 
    counter = 0
    num = lst[0]      
    for i in lst: 
        curr_frequency = lst.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num

activies = ['SB', 'ST', 'WO', 'WI', 'WA', 'ET', 'LB', 'NA', 'SD']
activity_dict = {'SB':0, 'ST':1, 'WO':2, 'WI':3, 'WA':4, 'ET':5, 'LB':6, 'NA':7, 'SD':8}

def one_hot_activities(activities):
    onehots = []
    for a in activities:
        onehot = [0,0,0,0,0,0,0,0,0]
        onehot[a-1]=1
        onehots.append(onehot)
    return onehots

def get_activity_labels_windows(activities,window_size = 50,window_step = 10):
    windows_activities=[]
    i=0
    while i + window_size < len(activities):
        window =  activities[i:i+window_size]
        i+=window_step
        windows_activities.append(most_frequent(window))
    return windows_activities

def get_measurements_windows(measurements_vectors,window_size = 50,window_step = 10):
    windows = []
    i=0
    while i + window_size < len(measurements_vectors):
        window = measurements_vectors[i:i+window_size]
        i+=window_step
        windows.append(window)
        if len(window) != window_size:
            print('not window size')
    return windows

def get_additional_features(measurement):
    len_r = math.sqrt(1.0*pow(measurement[0],2)+1.0*pow(measurement[2],2)+1.0*pow(measurement[4],2))
    len_p = math.sqrt(1.0*pow(measurement[1],2)+1.0*pow(measurement[3],2)+1.0*pow(measurement[5],2))
    alpha_r = math.acos(1.0*measurement[0]/len_r)
    beta_r = math.acos(measurement[2]/len_r)
    gama_r = math.acos(1.0*measurement[4]/len_r)
    alpha_p= math.acos(1.0*measurement[1]/len_p)
    beta_p= math.acos(1.0*measurement[3]/len_p)
    gama_p= math.acos(1.0*measurement[5]/len_p)
    return [len_r,len_p,alpha_r,beta_r,gama_r,alpha_p,beta_p,gama_p]

def read_data_from_file(filename):
    sensors_measurements = []
    activities = []
    with open('data\\'+filename) as file:
        line = file.readline()
        while line:
            parts = line.replace('\n','').split(',')
            measures = [int(i) for i in parts[0:-1]]
            measures.extend(get_additional_features(measures))
            sensors_measurements.append(measures)
            activities.append(activity_dict[parts[-1]]+1)
            line = file.readline()
    return (sensors_measurements,activities)

def read_data():
    sensors_measurements = []
    activities = []
    for filename in os.listdir("data"):
        ins,outs = read_data_from_file(filename)
        sensors_measurements.extend(ins)
        activities.extend(outs)
    return (sensors_measurements,activities)

measurements,activities = read_data()

windowsize = 50
windowstep = 25

activity_window_labels = get_activity_labels_windows(activities,window_size=windowsize,window_step=windowstep)
activities_onehot = one_hot_activities(activity_window_labels)

windows = get_measurements_windows(measurements,window_size=windowsize,window_step=windowstep)

sensors_windows_np = windows_np_arrays(windows)
train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, np.array(activity_window_labels), test_size=0.2, stratify=activity_window_labels)

scaled_train_in, scaled_test_in  = scale_train_test_sets(train_in,test_in)


# print(len(scaled_train_in))
# print(len(scaled_test_in))

# write_in_out_weka_style(scaled_train_in,train_out,fname='HAR_train_25')
# write_in_out_weka_style(scaled_test_in,test_out,fname='HAR_test_25')
# xsc = scaled_train_in
# xsc.extend(scaled_test_in)
# ysc=train_out.tolist()
# ysc.extend(test_out.tolist())
# write_in_out_weka_style(xsc,ysc,fname='HAR_50')

# get_features_for_windows(scaled_train_in)
# get_features_for_windows(scaled_test_in)
# write_sensor_measures_to_file(scaled_train_in,fname='train')
# write_sensor_measures_to_file(scaled_test_in,fname='test')
# write_features_to_file(scaled_train_in,train_out,fname='train')
# write_features_to_file(scaled_test_in,test_out,fname='test')

def split_indices_per_label(y):
    indicies_per_label = [[] for x in range(0,9)]
    # loop over the six labels
    for i in range(9):
        for li in range(0,len(y)):
            if y[li]-1 == i:
                indicies_per_label[i].append(li)
    return indicies_per_label

# def plot_cwt_coeffs_per_label(X, label_indicies, label_names, signal, sample, scales, wavelet):
    
#     fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(10,10))

#     for ax, indices, name in zip(axs.flat, label_indicies, label_names):
#         # if len(indices) <=sample:
#         #     continue
#         # apply  PyWavelets continuous wavelet transfromation function
#         data = X[indices[sample]]
#         series = [item[signal] for item in data]
#         coeffs, freqs = pywt.cwt(series, scales, wavelet = wavelet)
#         # create scalogram
#         ax.imshow(coeffs, cmap = 'coolwarm', aspect = 'auto')
#         ax.set_title(name)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.set_ylabel('Scale')
#         ax.set_xlabel('Time')
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig('wavelet_plots\\cwt\\'+SIGNAL_NAMES[signal]+'_64.png')

# # list of list of sample indicies per activity
# y=train_out.tolist()
# y.extend(test_out.tolist())
# X = scaled_train_in
# X.extend(scaled_test_in)

# train_labels_indicies = split_indices_per_label(y)
# for i in range(0,2):
#     print(LABEL_NAMES[i])
#     ind = train_labels_indicies[i][0]
#     print('0')
#     for x in [item[0] for item in X[ind]]:
#         print(x)
#     print('1')
#     for x in [item[1] for item in X[ind]]:
#         print(x)
#     print('2')
#     for x in [item[2] for item in X[ind]]:
#         print(x)
#     print('3')
#     for x in [item[3] for item in X[ind]]:
#         print(x)
#     print('4')
#     for x in [item[4] for item in X[ind]]:
#         print(x)
#     print('5')
#     for x in [item[5] for item in X[ind]]:
#         print(x)

signal = 1 # signal index
sample = 1 # sample index of each label indicies list
scales = np.arange(1, 65) # range of scales
wavelet = 'morl' # mother wavelet


# for signal in range(0,len(SIGNAL_NAMES)):
#     plot_cwt_coeffs_per_label(X, train_labels_indicies, LABEL_NAMES, signal, sample, scales, wavelet)