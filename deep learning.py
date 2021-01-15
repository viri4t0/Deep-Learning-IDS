# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:04:57 2019

@author: viri4t0
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from itertools import islice
import matplotlib.pyplot as plt


fd = open("test50.csv")
datos = []
objetivos = []
for linea in fd:
    cols = [float(x) for x in linea.split(",")[8:21]] #[7:83]
    objs = [linea.split(",")[87]]
    datos.append(cols)
    objetivos.append(objs)
fd.close()


#Preprocesando los datos
#SE NORMALIZAN LOS DATOS ENTRENAMIENTO
x_entrena = np.array(datos)
maximos = x_entrena.max(axis=0)
for fila in range(len(x_entrena)):
    for col in range(len(maximos)):
        x_entrena[fila,col] /= maximos[col]


#OHE de las clases etiquetadas
y_entrena = np.array(objetivos)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_entrena)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_entrena = onehot_encoder.fit_transform(integer_encoded)


#DIVIDIMOS LOS DATOS DE ENTRENAMIENTO 
X_train, X_test, y_train, y_test = train_test_split(x_entrena, y_entrena, test_size=0.20, random_state=42)


#CREAMOS EL MODELO
def create_model1():  
    model = Sequential()
    model.name="MLP1-Sin Dropout-RMSProp"
    model.add(Dense(250, activation='relu', input_shape =(13,), name="capa_oculta_1"))
    model.add(Dense(2, activation='softmax', name="Capa_de_Salida-softmax"))
    
    return model

def create_model2():  
    model = Sequential()
    model.name="MLP2-Sin Dropout-RMSProp"
    model.add(Dense(250, activation='relu', input_shape =(13,), name="capa_oculta_1"))
    model.add(Dense(250, activation='relu', name="capa_oculta_2"))
    model.add(Dense(250, activation='relu', name="capa_oculta_3"))
    model.add(Dense(2, activation='softmax', name="Capa_de_Salida-softmax"))
    
    return model

def create_model3():  
    model = Sequential()
    model.name="MLP3-Sin Dropout-Adam"
    model.add(Dense(250, activation='relu', input_shape =(13,), name="capa_oculta_1"))
    model.add(Dense(250, activation='relu', name="capa_oculta_2"))
    model.add(Dense(250, activation='relu', name="capa_oculta_3"))
    model.add(Dense(250, activation='relu', name="capa_oculta_4"))
    model.add(Dense(250, activation='relu', name="capa_oculta_5"))
    model.add(Dense(2, activation='softmax', name="Capa_de_Salida-softmax"))
    
    return model


# OPTIMIZADOR RMSPROP
modelo1 = create_model1()
modelo1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy','mse'])
modelo1Datos = modelo1.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=225, batch_size=128)

# OPTIMIZADOR ADAM
modelo2 = create_model2()
modelo2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy','mse'])
modelo2Datos = modelo2.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=225, batch_size=128)

# OPTIMIZADOR SGD
modelo3 = create_model3()
modelo3.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy','mse'])
modelo3Datos = modelo3.fit(X_train, y_train,validation_data=(X_test,y_test), epochs=225, batch_size=128)

modelo1.summary()
modelo2.summary()
modelo3.summary()


#GRÁFICAS
def crear_grafica_accuracy():
    plt.style.use('ggplot')
    #RMS
    plt.plot(modelo1Datos.history['accuracy'],'r')
    print("Valor máximo modelo 1: " + str(max(modelo1Datos.history['accuracy'])) )
    #ADAM
    plt.plot(modelo2Datos.history['accuracy'],'g')
    print("Valor máximo modelo 2: " + str(max(modelo2Datos.history['accuracy'])) )
    #SGD
    plt.plot(modelo3Datos.history['accuracy'],'b')  
    print("Valor máximo modelo 3: " + str(max(modelo3Datos.history['accuracy'])) )
    #CONFIG
    plt.xticks(np.arange(0, 226, 25.0))  
    plt.xlabel("Número de épocas")  
    plt.ylabel("Exactitud")  
    plt.title("Exactitud: RMSProp 1 capa vs RMSProp 2 capas vs Adam 3 capas")  
    plt.legend(['RMSProp 1 capa','RMSProp 2 capas','Adam 3 capas'])
    plt.show()
    
def crear_grafica_loss():
    plt.style.use('ggplot')
    #RMS
    plt.plot(modelo1Datos.history['mse'],'r')
    print("Valor máximo loss modelo 1: " + str(min(modelo1Datos.history['mse'])) )
    #ADAM
    plt.plot(modelo2Datos.history['mse'],'g')
    print("Valor máximo loss modelo 2: " + str(min(modelo2Datos.history['mse'])) )
    #SGD
    plt.plot(modelo3Datos.history['mse'],'b')  
    print("Valor máximo loss modelo 3: " + str(min(modelo3Datos.history['mse'])) )
    #CONFIG
    plt.xticks(np.arange(0, 226, 25.0))  
    plt.xlabel("Número de épocas")  
    plt.ylabel("Error cuadrático médio")  
    plt.title("Error: RMSProp 1 capa vs RMSProp 2 capas vs Adam 3 capas")  
    plt.legend(['RMSProp 1 capa','RMSProp 2 capas','Adam 3 capas'])
    plt.show()

def crear_curvas_ROC():
    plt.style.use('ggplot')
    y_testMax = np.argmax(y_test, axis=-1)
    #RMS
    pred = np.argmax(modelo1.predict(X_test), axis=-1)
    fpr, tpr, _ = roc_curve(y_testMax, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,'r',label='RMSProp 1 capa: curva ROC (area = %0.3f)' % roc_auc)
    #ADAM
    pred = np.argmax(modelo2.predict(X_test), axis=-1)
    fpr, tpr, _ = roc_curve(y_testMax, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,'g', label='RMSProp 2 capas: curva ROC (area = %0.3f)' % roc_auc)
    #SGD
    pred = np.argmax(modelo3.predict(X_test), axis=-1)
    fpr, tpr, _ = roc_curve(y_testMax, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,'b', label='Adam 3 capas: curva ROC (area = %0.3f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curvas ROC: RMSProp 1 capa vs RMSProp 2 capas vs Adam 3 capas')
    plt.legend(loc="lower right")
    plt.show()


def print_confusion_matrix(cm):
    print('True negative = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True positive = ', cm[1][1])
    
def realizar_matriz():
    snn_pred = modelo1.predict(X_test, verbose=1)  
    snn_predicted = np.argmax(snn_pred, axis=1)  
    snn_cm = confusion_matrix(np.argmax(y_test, axis=1), snn_predicted)
    print(" modelo 1: falsos positivos etc")
    print_confusion_matrix(snn_cm)
    
    snn_pred = modelo2.predict(X_test, verbose=1)  
    snn_predicted = np.argmax(snn_pred, axis=1)  
    snn_cm = confusion_matrix(np.argmax(y_test, axis=1), snn_predicted)
    print(" modelo 2: falsos positivos etc")
    print_confusion_matrix(snn_cm)
    
    snn_pred = modelo3.predict(X_test, verbose=1)  
    snn_predicted = np.argmax(snn_pred, axis=1)  
    snn_cm = confusion_matrix(np.argmax(y_test, axis=1), snn_predicted)
    print(" modelo 3: falsos positivos etc")
    print_confusion_matrix(snn_cm)
    

crear_grafica_accuracy()
crear_grafica_loss()
crear_curvas_ROC()
realizar_matriz()
