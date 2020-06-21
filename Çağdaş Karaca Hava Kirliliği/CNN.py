import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train=pd.read_csv("D:/4. sınıf dersler/YMH418 Güncel Konular/Çağdaş Karaca Hava Kirliliği/yapaysiniragi/data/i_Detaylar19617703.04.2020_18_19_52.csv")

label_encoder=LabelEncoder().fit(train.Date)
labels=label_encoder.transform(train.Date)
classes=list(label_encoder.classes_)

train=train.drop(["Date","state"],axis=1)
test=labels
nb_features=4
nb_classes=len(classes)

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler().fit(train.values)
train=scaler.transform(train.values)

from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(train,labels,test_size=0.1)

from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train)
y_valid=to_categorical(y_valid)

X_train=np.array(X_train).reshape(23291, 4,1)
X_valid=np.array(X_valid).reshape(2588, 4,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv1D,Dropout,MaxPooling1D,Flatten

model=Sequential();

model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256,1))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X_train,y_train,epochs=15,validation_data=(X_valid,y_valid))

print("ortalama eğitim kaybı",np.mean(model.history.history["loss"]))
print("ortalama eğitim başarısı",np.mean(model.history.history["accuracy"]))
print("ortalama doğrulama kaybı",np.mean(model.history.history["val_loss"]))
print("ortalama doğrulama başarısı",np.mean(model.history.history["val_accuracy"]))

import matplotlib.pyplot as plt

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(15,15))
ax1.plot(model.history.history['loss'],color='g',label="Eğitim Kaybı")
ax1.plot(model.history.history['val_loss'],color='y',label="Doğrulama Kaybı")
ax1.set_xticks(np.arange(20,100,20))
ax2.plot(model.history.history['accuracy'],color='b',label="Eğitim Başarımı")
ax2.plot(model.history.history['val_accuracy'],color='r',label="Doğrulama Başarımı")
ax2.set_xticks(np.arange(20,100,20))

plt.legend()
plt.show
