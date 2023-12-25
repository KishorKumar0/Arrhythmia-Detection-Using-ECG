import numpy as np
import glob
import pywt
import matplotlib.pyplot as plt

def sliding_window(data,window_size):
    import numpy as np
    shape_data = (data.shape[-1]-window_size+1,window_size)
    steps=data.strides+(data.strides[-1],)
    win=np.lib.stride_tricks.as_strided(data,shape=shape_data,strides=steps)
    return win

def moving_mean(data,sampling_rate,window_size):
    import numpy as np
    data_avg=(np.mean(data))
    data_array=np.array(data)
    window = window_size*sampling_rate
    rolling_mean = np.mean(sliding_window(data_array,int(window)),axis=1)
    return rolling_mean

def detect_peak(data,sampling_rate,window_size):
    import numpy as np
    moving_avg= moving_mean(data,sampling_rate=sampling_rate,window_size=window_size)
    detection_fit= 20
    means = np.array(moving_avg)
    minimum = np.mean(means/100)*detection_fit
    moving_avg = means + minimum
    #print(moving_avg.shape)
    
    peaks_x = np.where((data>moving_avg))[0]
    peaks_y = data[np.where(data>moving_avg)[0]]
    
    peak_edges= np.where(np.diff(peaks_x)>1)[0]
    
    peak_edges= np.concatenate((np.array([0]),(np.where(np.diff(peaks_x)>1)[0]),np.array([len(peaks_x)])))
    
    peak_list = []
    
    for i in range (0,len(peak_edges)):
        try:
            y_values = peaks_y[peak_edges[i]:peak_edges[i+1]].tolist()
            peak_list.append(peaks_x[peak_edges[i] + y_values.index(max(y_values))])
        except:
            pass
    return peak_list

def calc_rrinterval(data,sampling_rate,window_size):
    import numpy as np
    peak_list = detect_peak(data,sampling_rate=sampling_rate,window_size=window_size)
    peak_peak_list=(np.diff(peak_list)/sampling_rate)*1000
    dict_data={}
    dict_data['RR_intervals'] = peak_peak_list
    
    return dict_data

a=np.zeros((15000,100))
path=glob.glob('K:/chrome/kumar/files/*.csv')
j=0
for file in path:
    i=0
    fhand=open(file)
    for line in fhand:
        line=line.rstrip()
        words=line.split(',')
        a[i][j]=words[1]
        i+=1
    j+=1
b=a[:15000] 
b1=[]
for i in range(100):
    b2=b[:,i].reshape(1,15000)
    b1.append(b2)
b1=np.array(b1)
b1=b1.reshape(-1, b1.shape[-1])


df=b1
levels=8
RR_intervals=[]
for i in range(100):


    coeffs = pywt.wavedec(df[i], 'db6', level=8)
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    coeffs2 = [np.zeros(cA8.shape), cD8, cD7, cD6, cD5, np.zeros(cD4.shape), np.zeros(cD3.shape), np.zeros(cD2.shape), np.zeros(cD1.shape)]
    rec = pywt.waverec(coeffs2, 'db6')
   
   
    for i in range(100):
        value=calc_rrinterval(rec,250,60)
    RR_intervals.append(value)
        
peaks=detect_peak(rec, 250, 60)
    
fig, ax = plt.subplots(2, sharex=True, figsize=(12, 6))
fig.suptitle('Comparison of Original Signal and Detected Peaks')


ax[0].plot(df[i])
ax[0].set_title('Original Signal')


ax[1].plot(rec)
ax[1].set_title('Reconstructed Signal with Detected Peaks')
ax[1].plot(peaks, rec[peaks], 'x')
plt.show()






