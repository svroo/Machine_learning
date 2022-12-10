# from practica de k pliegues
from sklearn.model_selection import KFold
import numpy as np
import sys
import pickle
import csv
import pandas as pd
from sklearn.model_selection import train_test_split


class validation_set:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
class test_set:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

class data_set:
    def __init__(self, validation_set, test_set):
        self.validation_set = validation_set
        self.test_set = test_set

def create_csv(name_file, data, col_names, list_opt = False):
    new_data = data.tolist()

    with open(name_file, 'w', newline='') as f:
        if list_opt:
            new_new_data = [[i] for i in new_data]
        else:
            new_new_data = new_data
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(col_names)
        write.writerows(new_new_data)
        
def pliegues_validacion(file_name, pliegues, folder, test_size, mezcla, random_st):
    """
    Funcion para generar pliegues dependiendo del número de pliegues que se mandan a la función
    file_name : nombre del archivo a leer, o la ruta del mismo, preferentemente en forma dinamica
    pliegues : cantidad de pliegues a generar, numero entero
    folder : carpeta destino donde se van a a guardar los archivos generados por esta función
    test_size : tamaño de prueba
    mezcla : Valor booleano para mezclar o no los datos
    random_st : tamaño de la semilla para la mezcla
    """
    df = pd.read_csv(file_name, sep = ',', engine = 'python')
    columns_names = list(df.columns)
    n = len(columns_names) -1

    # corpus
    X = df[df.columns[0:n].values]
    # target
    y = df[str(columns_names[n])]

    #Separa corpus en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = mezcla, random_state=random_st)

    validation_sets = []

    # Número de pliegues
    kf = KFold(n_splits=pliegues)
    c = 0

    for train_index, test_index in kf.split(X_train):
        c = c + 1
        i = 1
        X_train_v, X_test_v = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_v, y_test_v = y_train.iloc[train_index], y_train.iloc[test_index]

        #Agrega el pliegue creado a la lista
        validation_sets.append(validation_set(X_train_v, y_train_v, X_test_v, y_test_v))

        #Almacena el conjunto de prueba
        my_test_set = test_set(X_test, y_test)	

        #Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
        my_data_set = data_set(validation_sets, my_test_set)

        my_data_set.test_set.X_test.to_csv(folder + 'data_test.csv', index = False)
        my_data_set.test_set.y_test.to_csv(folder + 'target_test.csv', index = False)

        cad_pliegues = str(pliegues) + '_'

        for val_set in my_data_set.validation_set:
            cad_i = str(i)
            val_set.X_train.to_csv(folder + 'data_validation_train_'+ cad_pliegues + cad_i + '.csv', index = False)
            val_set.y_train.to_csv(folder + 'target_validation_train_'+ cad_pliegues + cad_i+ '.csv', index = False)
            val_set.X_test.to_csv(folder + 'data_test_'+ cad_pliegues + cad_i+ '.csv', index = False)
            val_set.y_test.to_csv(folder + 'target_test_'+ cad_pliegues + cad_i + '.csv', index = False)
            i = i + 1
    print(f'Terminado los {pliegues} pliegues')
    return y_test