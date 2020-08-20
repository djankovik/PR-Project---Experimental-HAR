import numpy as np
from read_process_data import *
from utils import *
from model_builder import *
from sklearn.model_selection import train_test_split
from operator import itemgetter
import sys

# f = open("logs\\models_log1_ffts.txt", 'w')
# sys.stdout = f
# print('BEGIN')
results_metrics_list = []

#try for 1 - 8 second sized windows
for (windowsize,windowstep) in [(13,6),(25,12),(50,25),(75,37),(100,50),(125,62),(150,75)]:#,(175,87),(200,100),(225,112),(250,125)]:
    print(str(windowsize)+' - '+str(windowstep))
    activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
    activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

    sensor_measurements_windows_raw = get_sensor_measurements_sliding_windows([sensor_micro_L,sensor_pir_L,sensor_micro_R,sensor_pir_R,sensor_micro_U,sensor_pir_U],window_size=windowsize,window_step=windowstep)
    fft_window_components_dict = get_fft_elements_for_measurements_windows(sensor_measurements_windows_raw,np.fft.fft)
    sensors_windows_fft_np = windows_np_arrays(fft_window_components_dict['fft'])
    sensors_windows_amplitude_np = windows_np_arrays(fft_window_components_dict['amplitude'])
    sensors_windows_phase_np = windows_np_arrays(fft_window_components_dict['phase'])
    
    train_in, test_in, train_out, test_out = train_test_split(sensors_windows_fft_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
    results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="persensor_"+str(windowsize)+"\\"+str(windowstep)))

    train_in, test_in, train_out, test_out = train_test_split(sensors_windows_amplitude_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
    results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="persensor_"+str(windowsize)+"\\"+str(windowstep)))

    train_in, test_in, train_out, test_out = train_test_split(sensors_windows_phase_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
    results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="persensor_"+str(windowsize)+"\\"+str(windowstep)))


#sort a list of dictionaries by dictionary field
sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

print('********************Top 5 by Micro Avg P/R**************************')
print([i['name'] for i in sorted_macro_average_pr[0:5]])
for item in sorted_macro_average_pr[0:5]:
    for key in item.keys():
        print(key+": "+str(item[key]))
    print('\n')
    
print('********************Top 5 by Macro Avg P********************')
print([i['name'] for i in sorted_macro_average_p[0:5]])
# print([i for i in sorted_macro_average_p[0:5]])
for item in sorted_macro_average_p[0:5]:
    for key in item.keys():
        print(key+": "+str(item[key]))
    print('\n')
print('********************Top 5 by Macro Avg R********************')
print([i['name'] for i in sorted_macro_average_r[0:5]])
# print([i for i in sorted_macro_average_r[0:5]])
for item in sorted_macro_average_r[0:5]:
    for key in item.keys():
        print(key+": "+str(item[key]))
    print('\n')
print('********************Top 5 by Accuracy********************')
print([i['name'] for i in sorted_accuracy[0:5]])
# print([i for i in sorted_accuracy[0:5]])
for item in sorted_accuracy[0:5]:
    for key in item.keys():
        print(key+": "+str(item[key]))
    print('\n')
print('********************Top 5 by F1********************')
print([i['name'] for i in sorted_f1_avg[0:5]])
# print([i for i in sorted_f1_avg[0:5]])
for item in sorted_f1_avg[0:5]:
    for key in item.keys():
        print(key+": "+str(item[key]))
    print('\n')

# print('\nEND\n\n')
# f.close()


