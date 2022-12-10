from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
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
#     print(id_bueno)
    X[str(columns_names[n])].replace(X[str(columns_names[n])].values,id_bueno, inplace = True)

    return X

def clasificador_multinomial(X, y, i):
#     print ('\n------------Multinomial NB------------')
    clf = MultinomialNB()
    clf.fit(X, y.values.ravel())

#     target_names =clf.classes_
#     print (target_names)

    y_predict = clf.predict(X)
#     print (accuracy_score(y, y_predict))
    exactitud.append(accuracy_score(y, y_predict))
#     print (accuracy_score(y, y_predict, normalize=False))

#     print(classification_report(y, y_predict, target_names=target_names))
#     cm = confusion_matrix(y, y_predict, labels=target_names)
#     print (cm)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
#     disp.plot()
#     plt.show()
    print(f'Pliegue {i+1}, terminado')
    
def inicia_multi(path):
    # Creación de archivos para x, y
    name_files = os.listdir(path)
    data_train = list(filter(lambda x: x.startswith('data_validation_train'), name_files))
    target_train = list(filter(lambda x: x.startswith('target_validation_train'), name_files))
        
    for i, file in enumerate(data_train):
        x = pd.read_csv(path + file)
        x = preprocesamiento(x, 1)
        y = pd.read_csv(path + target_train[i])
        clasificador_multinomial(x, y, i)
    
    return exactitud

def calcula_promedio(exc = list()):
    """
    Funcion para calcular el promedio de un vector
    """
    n = len(exc)
    lista = np.array(exc)
    return np.sum(lista) / n

def multinomial_vs_test(path,y_test):
    """
    path : directorio donde se encuentran los pliegues previamente generados.
    test : valores de y sin dividir en pliegues
    """
    name_files = os.listdir(path)
    data_train = list(filter(lambda x: x.startswith('data_validation_train'), name_files))
        
    for i, file in enumerate(data_train):
        x = pd.read_csv(path + file)
        x = preprocesamiento(x, 1)        
        clasificador_multinomial(x, y_test, i)
        
    return exactitud

def get_tableau():
    """
    Función para mostrar en pantalla la tabla requerida en las evidencias.
    path : directorio donde se encuentran los k pliegues.
    """
    promedio = calcula_promedio(exactitud[:3])
    k_vs_t = calcula_promedio(exactitud[3:])
    
    vec = []
    vec.append([str('e_mail'), str('multinomial'), str(promedio), str(k_vs_t)])
    
    print(tabulate(vec,headers=['Dataset','Clasificador', 'Exactitud, promedio de los tres pliegues', 'Exactitud obtenida pliegues vs test'],tablefmt="grid", numalign="center"))