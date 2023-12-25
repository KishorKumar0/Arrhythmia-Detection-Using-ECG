import numpy as np
import pywt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import glob

def pantompkins(ecg, fs):
    lowcut = 5
    highcut = 15
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(1, [low, high], 'bandpass')
    ecg = filtfilt(b, a, ecg)

    
    diff_coeffs = np.array([1, 0, -1])
    ecg_diff = np.convolve(ecg, diff_coeffs)

    
    ecg_squared = ecg_diff ** 2

    window_size = int(0.15 * fs)
    window = np.ones(window_size)
    ecg_integral = np.convolve(ecg_squared, window)

    peaks, _ = find_peaks(ecg_integral, distance=int(0.5 * fs))

    rr_intervals = np.diff(peaks) / fs

    return peaks, rr_intervals


a= np.zeros((2500,100))
path = glob.glob('K:/chrome/kumar/files/*.csv')
j=0
for file in path:
    i=0
    fhand = open(file)
    for line in fhand:
        line = line.rstrip()
        words = line.split(',')
        a[i][j]=float(words[1]) 
        i+=1
    j+=1


signals = []
for i in range(100):
    signals.append(a[:, i])
    
rr_interval = []

for i in range(100):
    coeffs = pywt.wavedec(signals[i], 'db6', level=8)
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    coeffs2 = [np.zeros(cA8.shape), cD8, cD7, cD6, cD5, cD4, np.zeros(cD3.shape), np.zeros(cD2.shape), np.zeros(cD1.shape)]
    reconstructed_signal = pywt.waverec(coeffs2, 'db6')

    peaks, rr_intervals = pantompkins(reconstructed_signal, fs=250)

signals=signals[i]
peaks, rr_intervals = pantompkins(reconstructed_signal, fs=250)  
fig, ax = plt.subplots(2, sharex=True, figsize=(12, 6))
fig.suptitle('Comparison of Original Signal and Detected Peaks')

# Plot the original signal
ax[0].plot(signals[i])
ax[0].set_title('Original Signal')

#Plot the reconstructed signal and detected peaks
ax[1].plot(reconstructed_signal)
ax[1].set_title('Reconstructed Signal with Detected Peaks')
ax[1].plot(peaks-10, reconstructed_signal[peaks], 'x')
plt.show()
