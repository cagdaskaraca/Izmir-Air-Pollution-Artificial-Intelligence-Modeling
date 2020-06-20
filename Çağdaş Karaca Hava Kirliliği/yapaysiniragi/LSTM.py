import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,LSTM,BatchNormalization,Dropout,Flatten
from keras import backend as K

veri = pd.read_csv('D:/4. sınıf dersler/YMH418 Güncel Konular/Çağdaş Karaca Hava Kirliliği/yapaysiniragi/data/i_Detaylar19617703.04.2020_18_19_52.csv')
label_encoder=LabelEncoder().fit(veri.Date)
labels=label_encoder.transform(veri.Date)
classes=list(label_encoder.classes_)
veri=veri.drop(["Date","state"],axis=1)

nb_classes=len(classes)

scaler= StandardScaler().fit(veri.values)
veri=scaler.transform(veri.values)

X_train,X_valid,y_train,y_valid=train_test_split(veri,labels,test_size=0.2)

y_train=to_categorical(y_train)
y_valid=to_categorical(y_valid)

X_train=np.array(X_train).reshape(20703, 4,1)
X_valid=np.array(X_valid).reshape(5176, 4,1)


model=Sequential();

model.add(LSTM(512,input_shape=(4,1)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(2048,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="softmax"))
model.summary()
    

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

score=model.fit(X_train,y_train,epochs=5,validation_data=(X_valid,y_valid))

print(("Ortalama Eğitim Başarısı",np.mean(model.history.history["accuracy"])))
print(("Ortalama Doğrulama Başarısı",np.mean(model.history.history["val_accuracy"])))

import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'],color="b")
plt.plot(model.history.history['val_accuracy'],color="y")
plt.title("Model Başarımı")
plt.ylabel("Doğruluk")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show()

