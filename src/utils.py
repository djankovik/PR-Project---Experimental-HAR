import numpy as np

def pad_element_list(elementlists,padtosize=500,padwith=[0,0,0,0,0,0]):
    padded_lists = []
    for elementlist in elementlists:
        padded_list = []
        for element in elementlist:
            if len(padded_list) < padtosize:
                padded_list.append(element)
        while len(padded_list) < padtosize:
            padded_list.append(padwith)
        padded_lists.append(padded_list)
    return padded_lists

def one_hot_encode(targets,targets_vocabulary):
    onehots = []
    for target in targets:
        onehot = [0]*len(targets_vocabulary)
        onehot[targets_vocabulary.index(target)] = 1
        onehots.append(onehot)
    return onehots

def windows_np_arrays(windows):
    np_windows = []
    for window in windows:
        np_window = []
        for measurement in window:
            np_window.append(np.array(measurement))
        np_windows.append(np.array(np_window))
    return np.array(np_windows)

def make_np_arrays(arr):
    npified = []
    for a in arr:
        npified.append(np.array(a))
    return np.array(npified)

activity_vocab = ['SB', 'ST', 'WO', 'WI', 'WA', 'ET', 'LB', 'NA', 'SD']
def evaluate_model_predictions(predicted,expected,modelName = " "):
    dictcnt = {}
    dict_metrics = {}
    for activity in activity_vocab:
        dictcnt[activity] = {}
        dict_metrics[activity] = {}
        for act in activity_vocab:
            dictcnt[activity][act]= 0
        dictcnt[activity]['total'] = 0

    for (prediction,expectation) in zip(predicted,expected):
        exp = expectation.index(max(expectation))
        pred = prediction.tolist().index(max(prediction.tolist()))
        dictcnt[activity_vocab[exp]][activity_vocab[pred]] +=1
        dictcnt[activity_vocab[exp]]['total'] += 1

    dict_metrics['micro_average_p-r'] = 0.0
    dict_metrics['macro_average_p'] = 0.0
    dict_metrics['macro_average_r'] = 0.0
    dict_metrics['accuracy'] = 0.0
    dict_metrics['f1_avg'] = 0.0

    for activity in activity_vocab:
        are_really_activity = dictcnt[activity]['total']
        clasified_as_activity = 0
        for act in activity_vocab:
            clasified_as_activity += dictcnt[act][activity]
        
        if are_really_activity != 0:
            dict_metrics[activity]['precision'] = dictcnt[activity][activity]*1.0/are_really_activity
        else:
            dict_metrics[activity]['precision'] = 0.0
        if clasified_as_activity != 0:
            dict_metrics[activity]['recall'] = dictcnt[activity][activity]*1.0/clasified_as_activity
        else:
            dict_metrics[activity]['recall'] = 0.0
        
        if dict_metrics[activity]['recall'] != 0 or dict_metrics[activity]['precision'] != 0:
            dict_metrics[activity]['f1'] = 2.0*(dict_metrics[activity]['recall']* dict_metrics[activity]['precision'])/(dict_metrics[activity]['recall'] + dict_metrics[activity]['precision'])
        else:
            dict_metrics[activity]['f1'] = 0.0
        dict_metrics['macro_average_p'] += dict_metrics[activity]['precision']
        dict_metrics['macro_average_r'] += dict_metrics[activity]['recall']
        dict_metrics['micro_average_p-r'] += dictcnt[activity][activity]
        dict_metrics['accuracy'] += dictcnt[activity][activity]
        dict_metrics['f1_avg'] += dict_metrics[activity]['f1']
        
    dict_metrics['macro_average_p'] /= len(activity_vocab)
    dict_metrics['macro_average_r'] /= len(activity_vocab)
    dict_metrics['micro_average_p-r'] /= len(expected)
    dict_metrics['accuracy'] /= len(expected)
    dict_metrics['f1_avg'] /= len(activity_vocab)
    # print(dictcnt)

    # print("Model "+modelName+" report")
    # for key in dict_metrics.keys():
    #     print(key)
    #     print(dict_metrics[key])
    #     print(str(key)+"  ->  "+str(round(dict_metrics[key],3)))
    return dict_metrics



def get_window_trans_activities_label(window_activities):
    w_activities = list(set(window_activities))
    label = w_activities[0]
    for activity in w_activities[1:]:
        label = label +'_'+ activity
    return label

def get_activities_sliding_windows(activities,window_size = 50,window_step = 10,include_trans_activity = False,window_activity_label = 'dominant'):
    windows_activities=[]
    for i in range(0,len(activities)-window_size+1,window_step):
        window_activities =  activities[i:i+window_size]
        
        if include_trans_activity:
            windows_activities.append(get_window_trans_activities_label(window_activities))
            continue
        if window_activity_label == 'dominant':
            windows_activities.append(max(set(window_activities), key = window_activities.count))
        if window_activity_label == 'first':
            windows_activities.append(window_activities[0])
        if window_activity_label == 'last':
            windows_activities.append(window_activities[-1])

    return windows_activities

def get_transverse_vectors(measurements_vectors,window_size = 50,window_step = 10,include_trans_activity = False,window_activity_label = 'dominant'):
    moment_vectors = []
    for i in range(0,len(measurements_vectors[0])-window_size+1,window_step):
        window_vectors = []
        for vector in measurements_vectors:
            window_vectors.append(vector[i:i+window_size])
        moment_windows = []
        for i in range(0,window_size):
            window = []
            for vector in window_vectors:
                window.append(vector[i])
            moment_windows.append(window)
        moment_vectors.append(moment_windows)
    return moment_vectors


def get_sensor_measurements_sliding_windows(measurements_sensor_vector,window_size = 50,window_step = 10,include_trans_activity = False,window_activity_label = 'dominant'):
    windows = []
    for i in range(0,len(measurements_sensor_vector[0])-window_size+1,window_step):
        window = []
        for measurement in measurements_sensor_vector:
            window.append(measurement[i:i+window_size])
        windows.append(window)

    return windows
    

def get_fft_elements_for_measurements_windows(sensors_windows_measurements,fft_function):
    fft_windows = []
    amplitude_windows = []
    phase_windows = []
    reals_windows = []
    imags_windows = []

    for window in sensors_windows_measurements:
        window_ffts = []
        window_amplitudes = []
        window_phases = []
        window_reals = []
        window_imags = []
        for sensor_window in window:
            fft = fft_function(sensor_window)
            a=[]
            p=[]
            r=[]
            i=[]
            for c in fft:
                a.append(np.abs(c))
                p.append(np.angle(c))
                r.append(c.real)
                i.append(c.imag)
            window_ffts.append(fft)
            window_amplitudes.append(a)
            window_phases.append(p)
            window_reals.append(r)
            window_imags.append(i)
        fft_windows.append(window_ffts)
        amplitude_windows.append(window_amplitudes)
        phase_windows.append(window_phases)
        reals_windows.append(window_reals)
        imags_windows.append(window_imags)

    return {'fft':fft_windows,'amplitude':amplitude_windows,'phase':phase_windows,'real':reals_windows,'imaginary':imags_windows}

