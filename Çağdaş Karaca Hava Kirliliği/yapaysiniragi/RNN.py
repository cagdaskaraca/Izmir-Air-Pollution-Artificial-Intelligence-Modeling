import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

veri = pd.read_csv('D:/4. sınıf dersler/YMH418 Güncel Konular/Çağdaş Karaca Hava Kirliliği/yapaysiniragi/data/i_Detaylar19617703.04.2020_18_19_52.csv')


label_encoder=LabelEncoder().fit(veri.Date)
labels=label_encoder.transform(veri.Date)
classes=list(label_encoder.classes_)

x=veri.drop(["Date","state"],axis=1)
y=labels

nb_classes=len(classes)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

x_train=np.array(x_train).reshape(20703, 4,1)
x_test=np.array(x_test).reshape(5176, 4,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.layers import Flatten,SimpleRNN,BatchNormalization

model=Sequential()
model.add(SimpleRNN(512,input_shape=(4,1)))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2048,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="sigmoid"))
model.summary()



model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

score=model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))
print(" ")
print(" ")
print("ortalama doğrulama başarısı",np.mean(model.history.history["val_accuracy"]))
print("ortalama eğitim başarısı",np.mean(model.history.history["accuracy"]))

import matplotlib.pyplot as plt

plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Test"], loc="upper left")
plt.show()


