from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import zscore
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import numpy as np
from sklearn import metrics
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

path = "./data/"
filename_read = os.path.join(path,"i_Detaylar19617703.04.2020_18_19_52.csv")
df = pd.read_csv(filename_read)


def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

date = df['Date']
df.drop('Date',1,inplace=True)
state = df['state']
df.drop('state',1,inplace=True)
missing_median(df, 'so2')
missing_median(df, 'no2')
missing_median(df, 'rspm')
missing_median(df, 'spm')

dataset=df.values
x=dataset[:,0:12]
print(x)
y=dataset[:,2]
print(y)
kf = KFold(5) # K değerimiz

oos_y = []
oos_pred = []
fold = 0
for train, test in kf.split(x):
    fold+=1
    print("Fold #{}".format(fold))

x_train = x[train]
y_train = y[train]
x_test = x[test]
y_test = y[test]

model = Sequential()
model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1,activation="sigmoid"))
model.summary()


model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=1,epochs=6)

pred = model.predict(x_test)
oos_y.append(y_test)
oos_pred.append(pred)

score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Fold score (RMSE): {}".format(score))

oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
score = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
print("Final, out of sample score (RMSE): {}".format(score))



def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='Beklenen')
    plt.plot(t['pred'].tolist(), label='Tahmin')
    plt.title("Beklenen Tahmin")
    plt.ylabel('output')
    plt.legend()
    plt.show()

# Plot the chart
chart_regression(pred.flatten(),y_test)

print(("Ortalama Eğitim Başarısı",np.mean(model.history.history["accuracy"])))
print(("Ortalama Doğrulama Başarısı",np.mean(model.history.history["val_accuracy"])))
print(("Eğitim Kaybı",np.mean(model.history.history["loss"])))
print(("Ortalama Doğrulama kaybı",np.mean(model.history.history["val_loss"])))

import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'],color="b")
plt.plot(model.history.history['val_accuracy'],color="y")
plt.title("Model Başarımı")
plt.ylabel("Doğruluk")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show()

plt.plot(model.history.history['loss'],color="g")
plt.plot(model.history.history['val_loss'],color="r")
plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show()
