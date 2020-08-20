import re
import pandas as pd
import numpy as np
import zipfile
import os
import math
import statistics

def calculate_new_features(measurements):
    extended_msrmnts = []
    for measurement in measurements:
        len_r = math.sqrt(pow(measurement[0],2)+pow(measurement[2],2)+pow(measurement[4],2))
        len_p = math.sqrt(pow(measurement[1],2)+pow(measurement[3],2)+pow(measurement[5],2))
        alpha_r = math.acos(measurement[0]/len_r)
        beta_r = math.acos(measurement[2]/len_r)
        gama_r = math.acos(measurement[4]/len_r)
        alpha_p= math.acos(measurement[1]/len_p)
        beta_p= math.acos(measurement[3]/len_p)
        gama_p= math.acos(measurement[5]/len_p)
        avg_radar = 1.0*(measurement[0]+measurement[2]+measurement[4])/3
        avg_pir = 1.0*(measurement[1]+measurement[3]+measurement[5])/3
        sum_radar = measurement[0]+measurement[2]+measurement[4]
        sum_pir = measurement[1]+measurement[3]+measurement[5]
        extended_msrmnts.append([len_r,len_p,alpha_r,beta_r,gama_r,alpha_p,beta_p,gama_p,avg_radar,avg_pir,sum_radar,sum_pir])
    return extended_msrmnts

def get_normalized_measurements(sensors_measurements):
    normalized_measurements_sensors = []
    for i in range(0,len(sensors_measurements[0])):
        measurements = [item[i] for item in sensors_measurements]
        measurements_mean = np.mean(np.array(measurements).astype(np.float))
        measurements_std = np.std(np.array(measurements).astype(np.float))
        normalized_measurements = []
        for measurement in measurements:
            normal = ((float(measurement)-float(measurements_mean))/float(measurements_std))
            normalized_measurements.append(normal)
        normalized_measurements_sensors.append(normalized_measurements)
    return normalized_measurements_sensors


def read_data_from_file(filename):
    sensors_measurements = []
    activities = []
    with open('data\\'+filename) as file:
        line = file.readline()
        while line:
            parts = line.replace('\n','').split(',')
            measures = [int(i) for i in parts[0:-1]]
            sensors_measurements.append(measures)
            activities.append(parts[-1])
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

def get_data_as_measurement_series(sensors_measurements,activities):
    sensors_series = []
    activities_for_series = []
    current_series = []
    prev_activity = activities[0]

    for (measurements,activity) in zip(sensors_measurements,activities):
        if activity == prev_activity:
            current_series.append(measurements)
        else:
            sensors_series.append(current_series)
            activities_for_series.append(prev_activity)
            current_series = []
            prev_activity = activity
    return sensors_series,activities_for_series

def get_series_duration_stats(series_measurements):
    lengths = [len(item) for item in series_measurements]
    print(str(min(lengths))+", "+str(max(lengths))+", "+str(int(sum(lengths)/len(lengths))))
    return min(lengths), max(lengths), int(sum(lengths)/len(lengths))

def get_time_duration_seconds_for_each_activity(filename):
    activities_seconds = []
    with open(filename) as file:
        line = file.readline()
        while line:
            parts = re.split(r',', line.strip())
            from_ = parts[0]
            from_parts = from_.split(':')
            from_hrs = int(from_parts[0])
            from_min = int(from_parts[1])
            from_sec = int(from_parts[2])
            to_ = parts[1]
            to_parts = to_.split(':')
            to_hrs = int(to_parts[0])
            to_min = int(to_parts[1])
            to_sec = int(to_parts[2])

            activity_seconds_duration = (to_hrs*3600+to_min*60+to_sec)-(from_hrs*3600+from_min*60+from_sec)
            activity_seconds = [parts[2],activity_seconds_duration]
            activities_seconds.append(activity_seconds)
            line = file.readline()
    print(activities_seconds)
    return activities_seconds

def get_activity_duration_pairs(activities):
    activity_duration_pairs = []
    last_activity = activities[0]
    duration_activity = 0
    activity_duration_stats = {}

    for activity in activities:
        if activity not in activity_duration_stats:
            activity_duration_stats[activity] = {'shortest': 1000, 'longest':0, "count": 0, "total": 0,"average": 0.0}

        if activity == last_activity:
            duration_activity +=1
        else:
            activity_duration_pairs.append((last_activity,duration_activity))
            activity_duration_stats[last_activity]['count'] +=1
            activity_duration_stats[last_activity]['total'] +=duration_activity
            if activity_duration_stats[last_activity]['shortest'] > duration_activity:
                activity_duration_stats[last_activity]['shortest']=duration_activity
            if activity_duration_stats[last_activity]['longest'] < duration_activity:
                activity_duration_stats[last_activity]['longest']=duration_activity
            last_activity = activity
            duration_activity = 1

    for activity in activity_duration_stats.keys():
        activity_duration_stats[activity]['average'] = int(activity_duration_stats[activity]['total']/activity_duration_stats[activity]['count'])

    print(activity_duration_stats)
    return activity_duration_pairs,activity_duration_stats

activities_vocabulary = ['SB', 'ST', 'WO', 'WI', 'WA', 'ET', 'LB', 'NA', 'SD']

#AS NATURAL AS THEY COME
sensors_measurements,activities = read_data()

sensor_micro_L = [item[0] for item in sensors_measurements]
sensor_pir_L = [item[1] for item in sensors_measurements]
sensor_micro_R = [item[2] for item in sensors_measurements]
sensor_pir_R = [item[3] for item in sensors_measurements]
sensor_micro_U = [item[4] for item in sensors_measurements]
sensor_pir_U = [item[5] for item in sensors_measurements]

#GET NORMALIZED MEASUREMENTS
sensors_measurements_normalized = get_normalized_measurements(sensors_measurements)
sensor_micro_L_normalized = sensors_measurements_normalized[0]
sensor_pir_L_normalized = sensors_measurements_normalized[1]
sensor_micro_R_normalized = sensors_measurements_normalized[2]
sensor_pir_R_normalized = sensors_measurements_normalized[3]
sensor_micro_U_normalized = sensors_measurements_normalized[4]
sensor_pir_U_normalized = sensors_measurements_normalized[5]

#ADDITIONAL FEATURES CALCULATED
additional_features_measurements = calculate_new_features(sensors_measurements)
pir_vector_length_raw = [item[0] for item in additional_features_measurements]
micro_vector_length_raw = [item[1] for item in additional_features_measurements]
micro_alpha_raw = [item[2] for item in additional_features_measurements]
micro_beta_raw = [item[3] for item in additional_features_measurements]
micro_gama_raw = [item[4] for item in additional_features_measurements]
pir_alpha_raw= [item[5] for item in additional_features_measurements]
pir_beta_raw= [item[6] for item in additional_features_measurements]
pir_gama_raw= [item[7] for item in additional_features_measurements]
avg_micror_raw= [item[8] for item in additional_features_measurements]
avg_pir_raw= [item[9] for item in additional_features_measurements]
sum_micror_raw= [item[10] for item in additional_features_measurements]
sum_pir_raw= [item[11] for item in additional_features_measurements]

additional_features_measurements_normalized = get_normalized_measurements(additional_features_measurements)

pir_vector_length = additional_features_measurements_normalized[0]
micro_vector_length = additional_features_measurements_normalized[1]
micro_alpha = additional_features_measurements_normalized[2]
micro_beta = additional_features_measurements_normalized[3]
micro_gama = additional_features_measurements_normalized[4]
pir_alpha= additional_features_measurements_normalized[5]
pir_beta= additional_features_measurements_normalized[6]
pir_gama= additional_features_measurements_normalized[7]
avg_micror= additional_features_measurements_normalized[8]
avg_pir= additional_features_measurements_normalized[9]
sum_micror= additional_features_measurements_normalized[10]
sum_pir= additional_features_measurements_normalized[11]


def one_hot_encode(items,item_vocabulary):
    onehots = []
    for item in items:
        onehot = [0]*len(item_vocabulary)
        onehot[item_vocabulary.index(item)] = 1
        onehots.append(onehot)
    return onehots

activities_onehots = one_hot_encode(activities,activities_vocabulary)


#VECTOR FOR EACH MOMENT
# from utils import *
# moment_vectors = get_transverse_vectors([sensor_micro_L_normalized,sensor_pir_L_normalized,sensor_micro_R_normalized,sensor_pir_R_normalized,sensor_micro_U_normalized,sensor_pir_U_normalized])
# print(moment_vectors[5])