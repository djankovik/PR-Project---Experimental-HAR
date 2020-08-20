from utils import *

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Masking, Embedding, LSTM,TimeDistributed,  GRU, SimpleRNN, Bidirectional, Conv2D, Conv1D, MaxPooling2D,MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.vis_utils import plot_model
from ann_visualizer.visualize import ann_viz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def simpleDense1_model(train_in,test_in,train_out,test_out,inputshape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(inputshape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\simpleDense1.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),modelName="simpleDense")
    return evaluated

def simpleDense2_model(train_in,test_in,train_out,test_out,inputshape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(inputshape,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\simpleDense2.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(),modelName="simpleDense")
    return evaluated

##########################################################################################################################

def Conv2D_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(train_in.shape[1],train_in.shape[2],train_in.shape[3]),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(9, activation='softmax'))

    ##plot_model(model, to_file='modelplots\conv2D.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="conv2D")
    return evaluated

###################################################################################################
def simpleRNN1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(SimpleRNN(units=64, activation="relu",input_shape=(train_in.shape[1:])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\simpleRNN1.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="simpleRNN")
    return evaluated

def simpleRNN2_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(SimpleRNN(units=64, activation="relu"))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\simpleRNN2.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="simpleRNN")
    return evaluated

def simpleRNN_TimeDistributed_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(SimpleRNN(units=32, activation="relu",return_sequences=True,dropout=0.1, recurrent_dropout=0.1))
    model.add(TimeDistributed(Dense(32, activation='relu'))) 
    model.add(SimpleRNN(units=32, activation="relu"))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\simpleRNN_TD.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="simpleRNN")
    return evaluated

def simpleRNN_moreDense_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(SimpleRNN(units=100, activation="relu"))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\simpleRNN_dense.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="simpleRNN")
    return evaluated
##########################################################################################################################

def simpleGRU1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(GRU(64, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\GRU1.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="gru")
    return evaluated

def simpleGRU2_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(GRU(100, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\GRU2.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="gru")
    return evaluated

def simpleGRU_TimeDistributed1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(GRU(64, dropout_W=0.2, dropout_U=0.2, return_sequences = True))
    model.add(TimeDistributed(Dense(64, activation='relu'))) 
    model.add(GRU(64, dropout_W=0.2, dropout_U=0.2, return_sequences = False))
    model.add(Dropout(0.4))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\GRU_TD.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="gru")
    return evaluated

def simpleGRU_TimeDistributed2_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(GRU(64, dropout_W=0.2, dropout_U=0.2, return_sequences = True))
    model.add(TimeDistributed(Dense(64, activation='relu'))) 
    model.add(GRU(32, dropout_W=0.2, dropout_U=0.2, return_sequences = True))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(GRU(32, dropout_W=0.2, dropout_U=0.2, return_sequences = False))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\GRU_TD2.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, np.array(train_out), batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="gru")
    return evaluated

##########################################################################################################################
def multiLSTM_TimeDistributed1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(TimeDistributed(Dense(32, activation='relu')))  
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\multiLSTM_TD1.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="lstm_timedist_lst..")
    return evaluated

def multiLSTM_TimeDistributed2_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(TimeDistributed(Dense(32, activation='relu'))) 
    model.add(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
    model.add(TimeDistributed(Dense(128, activation='relu'))) 
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\multiLSTM_TD2.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="lstm_timedist_lst..")
    return evaluated

def multiLSTM_lessdense_TimeDistributed1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\multiLSTM_TD1_lessdense.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="lstm_timedist_lst..")
    return evaluated

##########################################################################################################################

def LSTM1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\LSTM1.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="simpleLSTM")
    return evaluated

def LSTM2_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\LSTM2.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="simpleLSTM")
    return evaluated

def LSTM3_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\LSTM3.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="simpleLSTM")
    return evaluated

##########################################################################################################################
def BILSTM1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\BILSTM1.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="bi->lstm")
    return evaluated

def BILSTM2_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\BILSTM2.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="bi->lstm")
    return evaluated

def BILSTMx2_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\BILSTMx2.png', show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="bi->lstm")
    return evaluated

def BILSTM_TimeDistributed1_model(train_in,test_in,train_out,test_out):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(Bidirectional(LSTM(32, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(9, activation='softmax'))
    #plot_model(model, to_file='modelplots\BILSTM_TD.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(train_in, train_out, batch_size=32, epochs=15, verbose=0) 
    y_pred = model.predict(test_in)
    evaluated = evaluate_model_predictions(y_pred,test_out.tolist(), modelName="bi->lstm")
    return evaluated

##########################################################################################################################
def build_train_test_simpleDense_models(train_in_raw,test_in_raw,train_out_raw,test_out_raw,name=""):
    train_out = np.array(train_out_raw)
    test_out = np.array(test_out_raw)
    train_in = make_np_arrays(train_in_raw)
    test_in = make_np_arrays(test_in_raw)

    metrics_results = []
    r = simpleDense1_model(train_in,test_in,train_out,test_out,inputshape=len(train_in_raw[0]))
    r['name'] = name + '_simpleDense1_model'
    metrics_results.append(r)

    r = simpleDense2_model(train_in,test_in,train_out,test_out,inputshape=len(train_in_raw[0]))
    r['name'] = name + '_simpleDense2_model'
    metrics_results.append(r)

    return metrics_results
    

def build_train_test_models(train_in_raw,test_in_raw,train_out_raw,test_out_raw,name=""):
    train_out = np.array(train_out_raw)
    test_out = np.array(test_out_raw)
    train_in = make_np_arrays(train_in_raw)
    test_in = make_np_arrays(test_in_raw)
    
    metrics_results = []
    #print('______'+name+'______')
    #Simple RNN
    r = simpleRNN1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleRNN1_model'
    metrics_results.append(r)
    r = simpleRNN2_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleRNN2_model'
    metrics_results.append(r)
    r = simpleRNN_moreDense_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleRNN_moreDense_model'
    metrics_results.append(r)
    r = simpleRNN_TimeDistributed_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleRNN_TimeDistributed_model'
    metrics_results.append(r)
    
    r = simpleGRU1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleGRU1_model'
    metrics_results.append(r)
    r = simpleGRU2_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleGRU2_model'
    metrics_results.append(r)
    r = simpleGRU_TimeDistributed1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleGRU_TimeDistributed1_model'
    metrics_results.append(r)
    r = simpleGRU_TimeDistributed2_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_simpleGRU_TimeDistributed1_model'
    metrics_results.append(r)

    r = multiLSTM_TimeDistributed1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_multiLSTM_TimeDistributed1_model'
    metrics_results.append(r)
    r = multiLSTM_TimeDistributed2_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_multiLSTM_TimeDistributed2_model'
    metrics_results.append(r)
    r = multiLSTM_lessdense_TimeDistributed1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_multiLSTM_lessdense_TimeDistributed1_model'
    metrics_results.append(r)

    r = LSTM1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_LSTM1_model'
    metrics_results.append(r)
    r = LSTM2_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_LSTM2_model'
    metrics_results.append(r)
    r = LSTM3_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_LSTM3_model'
    metrics_results.append(r)

    r = BILSTM1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_BILSTM1_model'
    metrics_results.append(r)
    r = BILSTM2_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_BILSTM2_model'
    metrics_results.append(r)
    r = BILSTMx2_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_BILSTMx2_model'
    metrics_results.append(r)
    r = BILSTM_TimeDistributed1_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_BILSTM_TimeDistributed_model'
    metrics_results.append(r)

    #Convolution additionl procesing
    rows =len(train_in_raw[0])
    columns =len(train_in_raw[0][0])
    train_in = train_in.reshape(-1, rows, columns, 1)
    test_in = test_in.reshape(-1, rows,columns, 1)
    # ##print(train_in.shape, test_in.shape)

    r = Conv2D_model(train_in,test_in,train_out,test_out)
    r['name'] = name + '_Conv2D'
    metrics_results.append(r)

    return metrics_results