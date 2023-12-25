import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pywt
from scipy.signal import find_peaks


with open('rr_intervals.pickle', 'rb') as f:
    rr_intervals = pickle.load(f)

# Convert the RR intervals to a numpy array
rr_intervals = np.array(rr_intervals)


# Load the expected output values
l =[]
ans =[]
file2 = open("answers.txt","r+")
for f in file2.readlines(799):
    l = f.split(',')
    ans.append(l[1][0:1])
rr=np.zeros((100,70))
for i in range(100):
    for j in range(70):
        rr[i][j]=rr_intervals[i][j]
        
rr1=list(rr)
rr2=np.expand_dims(rr1,2) 
le = LabelEncoder()
le.fit(ans)
ans_encoded = le.transform(ans)
print(rr_intervals.shape[0])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,1), activation='relu', input_shape=(70,1,1)),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(rr2, ans_encoded, epochs=200)
model.summary()


predicted_values = model.predict(rr2)
predicted_classes = np.round(predicted_values).flatten()
true_classes = ans_encoded
loss, accuracy = model.evaluate(rr2, ans_encoded)
print('Training accuracy:', accuracy)

'''a = np.zeros((15000, 50))
path = glob.glob('C:/Users/Kishor Kumar/test/*.csv')
j = 0
for file in path:
    i = 0
    fhand = open(file)
    for line in fhand:
        line = line.rstrip()
        words = line.split(',')
        a[i][j] = float(words[1]) 
        i += 1
    j += 1
    
signals = []
for i in range(50):
    signals.append(a[:, i])
rr_intervals = []

for i in range(50):
    coeffs = pywt.wavedec(signals[i], 'db6', level=8)
    cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    coeffs2 = [np.zeros(cA8.shape), cD8, cD7, cD6, cD5, np.zeros(cD4.shape), np.zeros(cD3.shape), np.zeros(cD2.shape), np.zeros(cD1.shape)]
    reconstructed_signal = pywt.waverec(coeffs2, 'db6')
    
    peaks, _ = find_peaks(reconstructed_signal, distance=50)
    rr_intervals.append(np.diff(peaks))

rr_intervals = np.array(rr_intervals)

rr = np.zeros((48, 70))
for i in range(48):
    for j in range(70):
        rr[i][j] = rr_intervals[i][j]
        
rr1 = list(rr)
rr3 = np.expand_dims(rr1, 1)
test_label = [0]*24 + [1]*24
test_label = np.array(test_label)

predicted_values = model.predict(rr3)
predicted_classes = np.round(predicted_values).flatten()
true_classes = test_label
print(predicted_classes )
loss, accuracy = model.evaluate(rr3, test_label)
print(accuracy)
'''


