import numpy as np
import pandas as pd

import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tabulate import tabulate

exactitud = list()


def preprocesamiento(X, columna= int()):
    """
    Función creada para reemplazar los valores de la primera columna, porque naive bayes no acepta string.
    X : Pliegue o dataset
    retorna el dataset modificado
    """
    columns_names = list(X.columns)
    n = 0
    
    if n > 0:
        n = columna -1
    
    id_bueno = list()
    for i in X[str(columns_names[n])]:
        g = i.split(' ')
        id_bueno.append(g[1])

    X[str(columns_names[n])].replace(X[str(columns_names[n])].values,id_bueno, inplace = True)

    return X

def calcula_gaussian(x, y, i, emails = False):
    """
    Funcion para crear el modelo y entrenarlo a partir de los datos de entrenamiento, retorna los datos predichos.
    x : Valores de x de entrenamiento
    y : Valores de y de entrenamiento
    """
    clf = GaussianNB()
    clf.fit(x, y.values.ravel())
        
    y_predict = clf.predict(x)

    exactitud.append(accuracy_score(y, y_predict))


    print (accuracy_score(y, y_predict, normalize=False))
    labels = []
    if emails:
        target_names = ['Yes', 'No']
        labels = [1, 0]
    else:
        target_names = clf.classes_
        labels = target_names

    print(classification_report(y, y_predict, target_names=target_names))
    print (confusion_matrix(y, y_predict, labels=labels))

    print(f'Pliegue {i+1}, terminado')
    
def inicializa_gaus(path = str(), pre = False, emails = False):
    """
    Función que a partir del directorio otorgado, genera dos listas con los nombres de los archivos de entrenamiento, tanto para x, como y.
    Regresa una lista con el porcentaje de exactitud de cada pliegue
    path : string, dirección, de preferencia dinamica, donde se encuentran los pliegues, previamente generados.
    """
    
    # Creación de archivos para x, y
    name_files = os.listdir(path)
    data_train = list(filter(lambda x: x.startswith('data_validation_train'), name_files))
    target_train = list(filter(lambda x: x.startswith('target_validation_train'), name_files))
        
    for i, file in enumerate(data_train):
        x = pd.read_csv(path + file)
        if pre:
            x = preprocesamiento(x, 1)
        y = pd.read_csv(path + target_train[i])
        
        calcula_gaussian(x, y, i, emails)  
        
    return exactitud

def calcula_promedio(exc = list()):
    """
    Funcion para calcular el promedio de un vector
    """
    n = len(exc)
    lista = np.array(exc)
    return np.sum(lista) / n

def kpliegues_vs_test(path, pre = False, emails = False):
    """
    path : directorio donde se encuentran los pliegues previamente generados.
    test : valores de y sin dividir en pliegues
    """

    x = pd.read_csv(path + 'Xtest.csv')
    y = pd.read_csv(path + 'ytest.csv')

    if pre:
        x = preprocesamiento(x, 1)
        for i, v in enumerate(y):
            if v == 0:
                y[i] = 'No'
            elif v == 1:
                y[i] = 'Yes'
    
    calcula_gaussian(x, y, 0, emails)

    return exactitud

def get_tableau(datasetName = str(), inicio = int(), n = int()):
    """
    Función para mostrar en pantalla la tabla requerida en las evidencias.
    path : directorio donde se encuentran los k pliegues.
    """
    promedio = calcula_promedio(exactitud[inicio:n])
    k_vs_t = calcula_promedio(exactitud[n: n+2])
#     print('Promedio:', promedio, '\npliegues vs test', k_vs_t)
    
    vec = []
    vec.append([datasetName, str('Naive Bayes'), str(promedio), str(k_vs_t)])
    
    print(tabulate(vec,headers=['Dataset','Clasificador', 'Exactitud, promedio de los tres pliegues', 'Exactitud obtenida pliegues vs test'],tablefmt="grid", numalign="center"))