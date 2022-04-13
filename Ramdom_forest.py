# импорт класса ветки дерева
from Node import Node

# импорт библиотеки для создания структур данных
import pandas as pd

# импорт библиотеки для обработки массивов
import numpy as np

# класс случайного леса
class Random_forest:
    '''
    Класс реализации алгоритма классификации на основе случайного леса
    '''
    
    def __init__(self,
                 X: pd.DataFrame,
                 label_col: str,
                 min_samples_split: int,
                 max_depth: int,
                 q_trees = 5,
                 rule_value_step = None):
        '''
        Объект класса случайного леса

        Parameters
        ----------
        X : pd.DataFrame
            Тренировочный набор данных.
        label_col : str
            Название столбца целевых меток.
        min_samples_split : int
            минимальное количество экземпляров в листе.
        max_depth : int
            максимальная глубина каждого дерева принятия решений.
        q_trees : TYPE, optional
            Количество деревьев в лесу. The default is 5.
        rule_value_step : int, optional
            шаг подбора значения правила. The default is None.

        Returns
        -------
        Объект класса случайный лес.

        '''
    
        self.rule_value_step = rule_value_step if rule_value_step else 0.25
        
        self.q_trees = q_trees
        
        self.X = X
        
        self.label_col = label_col
        
        self.min_samples_split = min_samples_split
        
        self.max_depth = max_depth
        
        # атрибут записи массива объектов выращенных деревьев
        self.forest = None
        
    def grow_forest(self):
        '''
        Метод для выращивания случайного леса
        '''
        df = self.X.copy()
        
        # Разделение теренировочного сета на равные части для определённого
        # количества деревьев
        split = np.array_split(df, self.q_trees)
        
        # массив для записи объектов класса дерево принятия решений
        trees = []
        
        # счётчик индекса очередного дерева
        counter = 0
        
        # цикл для прохода по подвыборкам разделённого датачсета
        for piece in split:
            
            # формирование очередного объекта класса дерево принятия решений
            trees.append(Node(X = piece, label_col = self.label_col,
                              min_samples_split= self.min_samples_split,
                              max_depth=self.max_depth,
                              rule_value_step = self.rule_value_step))
            
            # выращивание очередного дерева
            trees[counter].grow_tree()
            
            counter += 1
        
        # запись полученного леса в соответствующий атрибут
        self.forest = trees
        
    def prune_forest(self, X):
        '''
        Метод для подрезки деревьев в лесу

        Parameters
        ----------
        X : pd.DataFrame
            Development set.

        Returns
        -------
        None.

        '''
        
        df = X.copy()
        
        # для каждого дерева в лесу вызывается метод подрезки дерева из класса
        # Node
        for i in range(self.q_trees):
            self.forest[i].prun_tree(df)
            
    def predict(self, X):
        '''
        Метод для формирования предсказаний модели случайного леса
        
        Parameters
        ----------
        X : pd.DataFrame
            Массив, в котором нужно классифицировать экземпляры
            
        Returns
        -------
        массив с предсказаниями для входного датасета.
        '''
        df = X.copy()
        
        # формирование массива для записи предсказаний
        predictions = []
        
        # формирование предскзаний для каждого дерева в лесу с помощью метода
        # из класса Node
        for i in range(self.q_trees):
            predictions.append(self.forest[i].predict(df))
            
        # транспланирование матрицы предскзаний для записи в 1 строку
        # метод для одного и того же экземпляра
        mag_predictions = list(np.array(predictions).transpose())
                
        # Цикл для выбора мажоритарного значения предсказания
        for i in range(len(df)):
            
            # выбор из матрицы очередной строки с предсказаниями одного
            # экземпляра
            current_array = mag_predictions[i]
            
            # Переопределение текущей строки в качестве серии из Pandas для
            # вычисления мажоритарного класса
            S = pd.Series(current_array)
            
            # формирование строки с уникальными значениями меток
            labels = S.unique()
            
            # количество голосов за мажоритарный класс
            max_count = 0
            
            # цикл прохода по строке с уникальными метками
            for label in labels:
                # количество голосов за текуший класс
                current_count = len(S.loc[S == label])
                
                # если количество голосов за текущий класс больше, чем за
                # мажоритарный, то текущий класс становится мажоритарным
                if (current_count > max_count):
                    mag_predictions[i] = label
                    max_count = current_count
        return(mag_predictions)
        