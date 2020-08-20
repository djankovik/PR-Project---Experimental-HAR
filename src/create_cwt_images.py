from skimage.transform import resize
from read_HAR_data_dwt import *

def transpose(arrarrarr):
    transposed = []
    for arrarr in arrarrarr:
        x = []
        for i in range(0,len(arrarr[0])):
            x.append([item[i] for item in arrarr])
        transposed.append(x)
    return transposed

def deep_np_arrays(windows):
    np_windows = []
    for window in windows:
        np_window = []
        for row in window:
            np_window.append(np.array(row,dtype=object))
        np_windows.append(np.array(np_window,dtype=object))
    return np.array(np_windows,dtype=object)

X_train = deep_np_arrays(transpose(scaled_train_in))
print(len(X_train))
print(len(X_train[0]))
print(len(X_train[0][0]))

y_train = np.array([i-1 for i in train_out])
X_test = deep_np_arrays(transpose(scaled_test_in))
y_test = np.array([i-1 for i in test_out])

def create_cwt_images(X, n_scales, rescale_size, wavelet_name = "morl"):
    n_samples = len(X)#X.shape[0] 
    n_signals = len(X[0][0])#X.shape[2] 
    
    # range of scales from 1 to n_scales
    scales = np.arange(1, n_scales + 1) 
    
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_signals), dtype = 'float32')
    
    for sample in range(n_samples):
        if sample % 1000 == 0:
            print(sample)
        for signal in range(n_signals):
            serie = [item[signal] for item in X[sample]] #X[sample, :, signal]
            # continuous wavelet transform 
            coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
            # resize the 2D cwt coeffs
            rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode = 'constant')
            X_cwt[sample,:,:,signal] = rescale_coeffs
            
    return X_cwt
  
# amount of pixels in X and Y 
rescale_size = 64
# determine the max scale size
n_scales = 64

X_train_cwt = create_cwt_images(X_train, n_scales, rescale_size)
print(f"shapes (n_samples, x_img, y_img, z_img) of X_train_cwt: {X_train_cwt.shape}")
X_test_cwt = create_cwt_images(X_test, n_scales, rescale_size)
print(f"shapes (n_samples, x_img, y_img, z_img) of X_test_cwt: {X_test_cwt.shape}")

