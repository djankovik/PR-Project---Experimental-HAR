from sklearn.model_selection import train_test_split
from read_process_data import *
from utils import *
from model_builder import *
from operator import itemgetter
import sys

# f = open("logs\\models_log1.txt", 'w')
# sys.stdout = f
# print('BEGIN')
results_metrics_list = []

#try for 1 - 8 second sized windows
for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
    activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
    activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

    sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_R_normalized,sensor_pir_R_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized],window_size=windowsize,window_step=windowstep)
    sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
    train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
    results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="persensor_"+str(windowsize)+"\\"+str(windowstep)))

# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('END\n\n')
# f.close()


#############################################################################################################################################
# #[pir_vector_length,micro_vector_length,micro_alpha,micro_beta,micro_gama,pir_alpha,pir_beta,pir_gama,avg_micror,avg_pir,sum_micror,sum_pir]
# #With additional metrics
# #Different combinations
# f = open("logs\\models_log2.txt", 'w')
# sys.stdout = f
# print('BEGIN')
# results_metrics_list = []

# #try for 1 - 8 second sized windows
# for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
#     activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
#     activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_R_normalized,sensor_pir_R_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized,pir_vector_length,micro_vector_length,micro_alpha,micro_beta,micro_gama,pir_alpha,pir_beta,pir_gama,avg_micror,avg_pir,sum_micror,sum_pir],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="persensor+allextended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_R_normalized,sensor_pir_R_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized,pir_vector_length,micro_vector_length,micro_alpha,micro_beta,micro_gama,pir_alpha,pir_beta,pir_gama],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="persensor+diplomskaextended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_R_normalized,sensor_pir_R_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized,micro_alpha,micro_beta,micro_gama,pir_alpha,pir_beta,pir_gama],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="persensor+anglesextended_"+str(windowsize)+"\\"+str(windowstep)))


# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('\nEND\n\n')
# f.close()


#############################################################################################################################################
#[pir_vector_length,micro_vector_length,micro_alpha,micro_beta,micro_gama,pir_alpha,pir_beta,pir_gama,avg_micror,avg_pir,sum_micror,sum_pir]
#With additional metrics
#Different combinations
#3
# f = open("logs\\models_log3.txt", 'w')
# sys.stdout = f
# print('BEGIN')
# results_metrics_list = []

# #try for 1 - 8 second sized windows
# for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
#     activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
#     activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_pir_L_normalized,sensor_pir_R_normalized,sensor_pir_U_normalized],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="onlyPIR_notextended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_pir_L_normalized,sensor_pir_R_normalized,sensor_pir_U_normalized,pir_vector_length,pir_alpha,pir_beta,pir_gama,avg_pir,sum_pir],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="onlyPIR_extended_"+str(windowsize)+"\\"+str(windowstep)))



# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('\nEND\n\n')
# f.close()

#4
# f = open("logs\\models_log4.txt", 'w')
# sys.stdout = f
# print('BEGIN')
# results_metrics_list = []

# #try for 1 - 8 second sized windows
# for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
#     activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
#     activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_micro_R_normalized,sensor_micro_U_normalized],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="onlyMIR_notextended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_micro_R_normalized,sensor_micro_U_normalized,micro_vector_length,micro_alpha,micro_beta,micro_gama,avg_micror,sum_micror],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="onlyMIR_extended_"+str(windowsize)+"\\"+str(windowstep)))

# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('\nEND\n\n')
# f.close()

# #5
# f = open("logs\\models_log5.txt", 'w')
# sys.stdout = f
# print('BEGIN')
# results_metrics_list = []

# #try for 1 - 8 second sized windows
# for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
#     activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
#     activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_R_normalized,sensor_pir_R_normalized],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="LeftRight_notextended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_R_normalized,sensor_pir_R_normalized,pir_vector_length,micro_vector_length,micro_alpha,micro_beta,pir_alpha,pir_beta],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="LeftRight_extended_"+str(windowsize)+"\\"+str(windowstep)))



# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('\nEND\n\n')
# f.close()

# #6
# f = open("logs\\models_log6.txt", 'w')
# sys.stdout = f
# print('BEGIN')
# results_metrics_list = []

# #try for 1 - 8 second sized windows
# for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
#     activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
#     activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="LeftUp_notextended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized,pir_vector_length,micro_vector_length,micro_alpha,micro_gama,pir_alpha,pir_gama],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="LeftUp_extended_"+str(windowsize)+"\\"+str(windowstep)))


# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('\nEND\n\n')
# f.close()

# #7
# f = open("logs\\models_log7.txt", 'w')
# sys.stdout = f
# print('BEGIN')
# results_metrics_list = []

# #try for 1 - 8 second sized windows
# for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
#     activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
#     activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_R_normalized,sensor_pir_R_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="RightUp_notextended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_R_normalized,sensor_pir_R_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized,pir_vector_length,micro_vector_length,micro_beta,micro_gama,pir_beta,pir_gama],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="RightUp_extended_"+str(windowsize)+"\\"+str(windowstep)))


# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('\nEND\n\n')
# f.close()

#8
f = open("logs\\models_log8.txt", 'w')
sys.stdout = f
print('BEGIN')
results_metrics_list = []

#try for 1 - 8 second sized windows
# for (windowsize,windowstep) in [(25,12),(50,25),(75,37),(100,50),(125,62),(150,75),(175,87),(200,100),(225,112),(250,125)]:
#     activity_windows= get_activities_sliding_windows(activities,window_size=windowsize,window_step=windowstep)
#     activities_onehot = one_hot_encode(activity_windows,activities_vocabulary)

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_U_normalized,sensor_pir_U_normalized,pir_vector_length,micro_vector_length,micro_gama,pir_gama],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="onlyUp_extended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_R_normalized,sensor_pir_R_normalized,pir_vector_length,micro_vector_length,micro_beta,pir_beta],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="onlyRight_extended_"+str(windowsize)+"\\"+str(windowstep)))

#     sensor_measurements_windows = get_sensor_measurements_sliding_windows([sensor_micro_L_normalized,sensor_pir_L_normalized,pir_vector_length,micro_vector_length,micro_alpha,pir_alpha],window_size=windowsize,window_step=windowstep)
#     sensors_windows_np = windows_np_arrays(sensor_measurements_windows)
#     train_in, test_in, train_out, test_out = train_test_split(sensors_windows_np, make_np_arrays(activities_onehot), test_size=0.33, random_state=42)
#     results_metrics_list.extend(build_train_test_models(train_in,test_in,train_out,test_out,name="onlyLeft_extended_"+str(windowsize)+"\\"+str(windowstep)))


# #sort a list of dictionaries by dictionary field
# sorted_macro_average_pr = sorted(results_metrics_list, key=itemgetter('micro_average_p-r'), reverse=True)
# sorted_macro_average_p = sorted(results_metrics_list, key=itemgetter('macro_average_p'), reverse=True)
# sorted_macro_average_r = sorted(results_metrics_list, key=itemgetter('macro_average_r'), reverse=True)
# sorted_accuracy = sorted(results_metrics_list, key=itemgetter('accuracy'), reverse=True)
# sorted_f1_avg = sorted(results_metrics_list, key=itemgetter('f1_avg'), reverse=True)

# print('********************Top 5 by Micro Avg P/R**************************')
# print([i['name'] for i in sorted_macro_average_pr[0:5]])
# for item in sorted_macro_average_pr[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
    
# print('********************Top 5 by Macro Avg P********************')
# print([i['name'] for i in sorted_macro_average_p[0:5]])
# # print([i for i in sorted_macro_average_p[0:5]])
# for item in sorted_macro_average_p[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Macro Avg R********************')
# print([i['name'] for i in sorted_macro_average_r[0:5]])
# # print([i for i in sorted_macro_average_r[0:5]])
# for item in sorted_macro_average_r[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by Accuracy********************')
# print([i['name'] for i in sorted_accuracy[0:5]])
# # print([i for i in sorted_accuracy[0:5]])
# for item in sorted_accuracy[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')
# print('********************Top 5 by F1********************')
# print([i['name'] for i in sorted_f1_avg[0:5]])
# # print([i for i in sorted_f1_avg[0:5]])
# for item in sorted_f1_avg[0:5]:
#     for key in item.keys():
#         print(key+": "+str(item[key]))
#     print('\n')

# print('\nEND\n\n')
# f.close()