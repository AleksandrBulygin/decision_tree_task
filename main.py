'''
Проект с реализацией алгоритма классификации на основе дерева принятия решений

Выполнил: Булыгин Александр
'''

# импорт библиотеки для создания структур данных
import pandas as pd

# импорт библиотеки для обработки массивов
import numpy as np

# импорт ниструмента для предобработки данных (нормировки)
from sklearn import preprocessing


# импорт класса ветки дерева
from Node import Node

# импорт инструмента для оценки качества классификации
from sklearn.metrics import f1_score

# импорт библиотеки для построения графиков
import matplotlib.pyplot as plt

# импорт модуля случайного леса
from Ramdom_forest import Random_forest

# импорт модуля реализации дерева принятия решений из библиотеки SciKit Learn
from sklearn.tree import DecisionTreeClassifier

def main():
    '''
    Создание DataFrame из датасета, используемого для классификации состояния
    шкалы баланса
    
    Атрибуты:
        LW - вес левой чаши;
        LD - длина левого плеча;
        RW - вес правой чаши;
        RD - длина правого плеча;
    
    Метки:
        B - стрелка шкалы весов уравновешена
        R - стрелка шаклы весов смещена вправо
        L - стрелка шкалы весов смещена влево
    '''
    df = pd.read_csv('balance-scale .data', 
                     sep = ',',
                     names = ['Class_name', 'LW', 'LD', 'RW', 'RD'])

    # запись имени столбца меток классов
    label_col = df.columns[0]
    
    
    '''
    Нормализация датасета по всем столбцам к диапазону от 0 до 1
    с помощью иструмента масштабирования MaxMinScaler
    '''
    
    df_copy = df.copy()
    scaler = preprocessing.MinMaxScaler()
    d = scaler.fit_transform(df_copy.drop([label_col], axis = 1))
    label_array = list(df_copy.pop(label_col).to_numpy())
    scaled_df = pd.DataFrame(d, columns = df.columns[1:])
    scaled_df.insert(0, label_col, label_array)
    
    '''
    перераспределение элементов датасета случайным образом для
    формирования репрезентативной выборки в тренировочный и тестовый датасеты
    '''
    scaled_df = scaled_df.sample(frac = 1)
    
    length_df = len(scaled_df)
    
    
    # Формирование тренировочного, тестового и dev наборов данных
    length_train_set = int(length_df*0.6)
    length_dev_set = int(length_df*0.2)
    
    train_set = scaled_df[:length_train_set]
    
    dev_set = scaled_df[length_train_set:(length_train_set + length_dev_set)]
    
    test_set = scaled_df[(length_train_set + length_dev_set):]
    
    # размерность квадратной матрицы подбора оптимальных параметров обучения
    # и рисования графиков
    det = 12
    
    # матрица для формирования массива зависимости метрики f1 от параметров
    # обучения для дерева принятия решений и случайного леса
    matrix = np.zeros([3, det, det])
    matrix_forest = np.zeros([3, det, det])
    
    # массив записи максимального значения f1 для каждого класса для дерева
    # принятия решений и случайного леса
    max_f1 = np.zeros(3)
    max_f1_forest = np.zeros(3)
    
    # массив для записи оптимального значения параметра минимального количества 
    # примеров из обучающей выборки, которые должны попасть при обучении в лист
    # для дерева приниятия решений и для случайного леса
    opt_samples = np.zeros(3)
    opt_samples_forest = np.zeros(3)
    
    # массив для записи оптимального значания параметра максимальной глубины
    # для дерева и для случайного леса
    opt_depth = np.zeros(3)
    opt_depth_forest = np.zeros(3)
    
    # количество деревьев в случайном лесу
    q_trees = 3
    
    # цикл для определения оптимальных параметров обучения
    for samples in range(1, det+1):
        for depth in range(det):
            
            # формирование объекта класса Node, который определён в файле
            # Node.py
            root = Node(X = train_set, label_col = label_col,
                        max_depth = depth, min_samples_split = samples)
            
            # выращивание дерева
            root.grow_tree()
            
            # подрезка дерева
            root.prun_tree(dev_set)
            
            # вычисление предсказаний классификации на основе дерева
            predictions_after = root.predict(test_set)
            
            # вычисление метрики f1 для каждого класса
            f1_after = f1_score(test_set[label_col].to_numpy(),
                                predictions_after, 
                                average = None)
            
            # формирование объекта класса Random_forest, который определён в
            # файле Ramdom_forest.py
            forest = Random_forest(X = train_set, label_col = label_col,
                                   min_samples_split = 
                                   samples,
                                   max_depth = depth,
                                   q_trees = q_trees)
            
            # выращивание леса
            forest.grow_forest()
            
            # подрезка деревьев леса (без подрезки классификация случайным
            # лесом лучше из-за того, что в наборе доля 1 из классов составляет
            # 8%)
            # forest.prune_forest(dev_set)
            
            # предсказания классов случайным лесом
            predictions_forest = forest.predict(test_set)
            
            # вычисление метрики f1 для каждого класса
            f1_forest = f1_score(test_set[label_col].to_numpy(),
                                predictions_forest, 
                                average = None)
            
            # запись очередных метрик классов в матрицы
            for i in range(3):
                matrix[i][samples - 1][depth] = f1_after[i]
                matrix_forest[i][samples - 1][depth] = f1_forest[i]
            
            # формирование логического массива условия, что очередное значение
            # метрики f1 для каждого класса больше каждого предыдущего
            max_f1_check = f1_after >= max_f1
            max_f1_check_forest = f1_forest >= max_f1_forest
            
            # массив для вычисления оптимальных параметров обучения на основе
            # полученного логического массива
            for i in range(3):
                if (max_f1_check[i] == True):
                    if (f1_after[i] > max_f1[i] or opt_samples[i] < samples):
                        opt_depth[i] = depth
                    max_f1[i] = f1_after[i]
                    opt_samples[i] = samples
                
                if (max_f1_check_forest[i] == True):
                    if(f1_forest[i] > max_f1_forest[i] or
                       opt_samples_forest[i] < samples):
                        opt_depth_forest[i] = depth
                    max_f1_forest[i] = f1_forest[i]
                    opt_samples_forest[i] = samples
    
    # массив меток классов
    labels = ['Balanced', 'Left', 'Right']
    
    # построение 3D графиков зависимости метрики f1 от параметров обучения
    x = np.linspace(1, det, det) 
    y = np.linspace(1, det, det) 
    X, Y = np.meshgrid(x, y)
    for i in range(3):
        ax = plt.axes(projection='3d')
        ax.set_title('Зависимость точности классификации от глубины дерева и'+
                     '\nминимального количества экземпляров в листе.'+
                     ' Класс - {}'.format(labels[i]) +'\nМаксимальное значение'
                     +' метрики достигается при:\n глубине:'+ 
                     ' {}\nминимальном количестве образцов: {}'.format(
                         opt_depth[i], opt_samples[i]) +
                     '\nмаксимальное значение f1 = {}'.format(max_f1[i]))
        ax.plot_surface(X, Y, matrix[i],
                        rstride=1, cstride=1, cmap='viridis')
        ax.set(xlabel='глубина дерева', 
           ylabel='минимальное количество образцов в листе', zlabel='f1')
        
        plt.show()
    
    
    for i in range(3):
        ax = plt.axes(projection='3d')
        ax.set_title('Зависимость точности классификации от глубины дерева и'+
                     '\nминимального количества экземпляров в листе.'+
                     'Случайный лес.\n Количество деревьев - {}.\n'.format(
                         q_trees)+
                     ' Класс - {}'.format(labels[i]) +'\nМаксимальное значение'
                     +' метрики достигается при:\n глубине:'+ 
                     ' {}\nминимальном количестве образцов: {}'.format(
                         opt_depth_forest[i], opt_samples_forest[i]) +
                     '\nмаксимальное значение f1 = {}'.format(
                         max_f1_forest[i]))
        ax.plot_surface(X, Y, matrix_forest[i],
                        rstride=1, cstride=1, cmap='viridis')
        ax.set(xlabel='глубина дерева', 
           ylabel='минимальное количество образцов в листе', zlabel='f1')
        
        plt.show()
    
    # формирование дерева с отимальными параметрами обучения
    # так как оптимальные параметры вычислены для каждого класса в отдельности
    # берётся среднее значение для каждого парметра
    root = Node(X = train_set, label_col = label_col,
                max_depth = int(np.mean(opt_depth)),
                min_samples_split = int(np.mean(opt_samples)))
    
    
    # выращивание дерева
    root.grow_tree()
    
    # подрезка дерева
    root.prun_tree(dev_set)
    
    # вычисление предсказаний классификации на основе дерева
    predictions = root.predict(test_set)
    
    # вычисление метрики f1 для каждого класса
    f1_opt = f1_score(test_set[label_col].to_numpy(),
                        predictions, 
                        average = None)
    
    # формирование объекта случайного леса с оптимальными параметрами обучения
    forest = Random_forest(X = train_set, label_col = label_col,
                           min_samples_split = 
                           int(np.mean(opt_samples_forest)), 
                           max_depth = int(np.mean(opt_depth_forest)),
                           q_trees = q_trees)
    
    # выращивание леса
    forest.grow_forest()
    
    # подрезка деревьев леса
    # forest.prune_forest(dev_set)
    
    # классификация по случайному лесу
    predictions_forest = forest.predict(test_set)
    
    # вычисление метрики f1 для каждого класса
    f1_forest_opt = f1_score(test_set[label_col].to_numpy(),
                        predictions_forest, 
                        average = None)
    
    # вывод в консоль метрик качества классификации для каждого класса
    print('f1 для каждого класса: ', f1_opt)
    print('f1 для каждого класса после классификации случайным лесом:',
          f1_forest_opt)
    

    # Создание объекта дерева принятия решений из библиотеки SciKt Learn
    clf = DecisionTreeClassifier()
    
    # столбец целевых меток тренировочного набора
    labels_train_set = train_set[label_col]
    
    # матрица атрибутов тренировочного набора
    features_train_set = train_set.drop(label_col, axis=1)
    
    # тренировка дерева
    clf.fit(features_train_set, labels_train_set)
    
    # матрица атрибутов тестового набора
    features_test_set = test_set.drop(label_col, axis=1)
    
    # вычисление предсказаний классификации
    predictions_SkLearn = clf.predict(features_test_set)
    
    # вычисление метрики f1 для каждого класса
    f1_SkLearn = f1_score(test_set[label_col].to_numpy(),
                        predictions_SkLearn, 
                        average = None)
    
    # вывод в консоль метрики качества классификации
    print("f1 для дерева, созданного из SkLearn: ", f1_SkLearn)
    
if __name__ == "__main__":
	main()