# библиотека для обработки массивов
import numpy as np

# библиотека для создания структур данных
import pandas as pd

# класс ветки дерева
class Node:
    '''
    Класс с реализацией алгоритма дерева принятия решений
    '''
    def __init__(
            
            self,
            X: pd.DataFrame,
            label_col: str,
            min_samples_split = None,
            max_depth = None,
            depth = None,
            node_type = None,
            rule = None,
            rule_value_step = None):
        '''
        Инициализация объекта класса дерева принятия решений, при создании
        объекта пользователем формирует корень дерева.
        
        атрибуты:
            X - тренировочный датасет;
            label_сol - название столбца меток класса,
            min_samples_split - минимальное количество экземпляров в листе;
            max_depth - максимальная глубина дерева;
            depth - текущая глубина дерева,
            node_type - тип узла дерева
            rule - логическое правило разделения в узле
            rule_value_step - шаг подбора значения правила
        '''
        
        self.X = X
        
        self.label_col = label_col
        
        self.min_samples_split = min_samples_split if min_samples_split else 7
        self.max_depth = max_depth if max_depth else 5
        
        self.rule_value_step = int(1/rule_value_step) if rule_value_step else 4
        
        self.depth = depth if depth else 0
        
        # счётчик использования узла при подрезке
        self.cnt_prun = 0
        
        # счётчик количество ошибок классификации при проходе через узел
        self.errors = 0
        
        # счётчик количества ошибок классификации в узле, как листе
        self.errors_as_Leaf = 0
        
        # список названий атрибутов
        self.features = list(self.X.columns[1:])
        
        # тип узла
        self.node_type = node_type if node_type else 'root'
        
        # массив с количеством экземпляров каждого класса во входном датасете
        self.counts = self.Class_counts(samples = self.X, Need = "count_dict",
                                      self = self)
        
        # критерий информативности Джини
        self.gini_criterion = self.GINI_criterion(samples = self.X,
                                                  self = self)
        
        # длина датасета
        self.n = len(X)
        
        # логическое привило
        self.rule = rule if rule else ""
        
        # правая и левая ветка узла
        self.left = None 
        self.right = None 

        # значения лучшего атрибута и его лучшего значения для формирования
        # правила узла
        self.best_feature = None 
        self.best_value = None
        
        # мажоритарный класс
        self.mag_class = None
        
        # вычисление мажоритарного класса
        if len(self.X) > 0:
            
            mag_class = self.Class_counts(samples = self.X, Need = "mag_class",
                                          self = self)
        self.mag_class = mag_class
    
        
    @staticmethod
    def GINI_criterion(samples, self):
        '''
        Метод вычисления критерия информативности Джини

        Parameters
        ----------
        samples : pd.DataFrame
            датасет.

        Returns
        -------
        float
            Критерий иноформативности Джини.

        '''
        
        # массив с количеством экземпляров каждого класса во входном массиве
        counts = self.Class_counts(samples = samples, Need = "counts",
                                   self = self)
        # количество экземпляров во входном массиве 
        n = sum(counts)
        
        if n == 0:
            return 0.0
        
        # вычисление вероятности случайного выбора экземпляра каждого класса
        # из массива
        p = np.array(counts)/n
        
        # вычисление критерия информативности Джини
        gini = 1 - sum(p**2)
        
        return gini
    
    @staticmethod
    def Class_counts(samples, Need, self):
        '''
        Метод для определения мажоритарного класса в массиве и количества
        экземпляров каждого класса в массиве

        Parameters
        ----------
        samples : pd.DataFrame
            датасет.
        Need : str
            аргумент необходимого от функции параметра:
                counts - массив количесва вхождений каждого класса в массив;
                mag_class - метка мажоритарного класса
                count_dict - словарь количества вхождений каждого класса в
                массив, в котором ключём является метка класса.

        Returns
        -------
        array or int or srt
            в зависимости от входного параметра Need возвращает:
                counts - массив количесва вхождений каждого класса в массив;
                mag_class - метка мажоритарного класса
                count_dict - словарь количества вхождений каждого класса в
                массив, в котором ключём является метка класса.
        '''
        
        # формирование массива уникальных меток класов в датасете
        class_labels =  list(samples[self.label_col].unique())
        
        # массив количесва вхождений каждого класса в массив
        counts = []
        
        # словарь количества вхождений каждого класса в
        # массив, в котором ключём является метка класса
        dict_class = {}
        
        # счётчик мажоритарного класса
        max_count = 0
        
        # цикл для высичления мажоритарного класса, массива и словаря
        # количества вхождений каждого класса в датасет
        for label in class_labels:
            
            # вычисление количества вхождений экземпляров очередного класса
            # в датасет
            counts_class = len(samples.loc[samples[self.label_col] == label])
            counts.append(counts_class)
            
            dict_class[label] = counts_class
            if (counts_class > max_count):
                max_count = counts_class
                mag_label = label
        
        if(Need == "counts"):
            return counts
        
        if(Need == "mag_class"):
            return mag_label
        
        if(Need == "count_dict"):
            return dict_class
            
            
    
    def best_Split(self):
        '''
        Метод для вычисления наилучшего разделения обучающей выборки
        
        Returns
        -------
        best_feature: int or str
            лучший атрибут для правила разделения
        best_value: float
            лучшее значение этого атрибута для правила
        '''
        df = self.X.copy()
        
        # если в тренировочном датасете узла все экземпляры одного класса,
        # то ничего не возвращать
        if(len(df[self.label_col].unique()) == 1):
            return(None, None)
        
        # Вычисление критерия Джини для датасета
        GINI_base = self.GINI_criterion(samples = df, self = self)
        
        # максимальный прирост информативности
        max_gain = 0
        
        # лучший атрибут для правила разделения
        best_feature = None
        
        # лучшее значение этого атрибута для правила
        best_value = None
        
        # цикл для прохода по каждому атрибуту
        for feature in self.features:
            
            # массив значений от 0 до 1/(шаг значения логического правилая)
            rules_predicates = range(self.rule_value_step + 1)
            
            # цикл для прохода по каждому значению логического правила
            for value in rules_predicates:
                
                # формирование выборок для правого и левого дочернего узла
                left_counts = df[df[feature] <= value/self.rule_value_step]
                right_counts = df[df[feature] > value/self.rule_value_step]
                
                # вычисление критерия Джини для левой и правой выборок
                gini_left = self.GINI_criterion(samples = left_counts,
                                                self = self)
                gini_right = self.GINI_criterion(samples = right_counts,
                                                 self = self)
                
                # определение вероятности перехода при классификации
                # экземпляра из тренировочного набора данных в правую ветку
                q0 = len(right_counts)/len(df)
                
                # определение взвешенного критерия Джини для правой и левой
                # выборки
                Pv = q0 * gini_right + (1- q0) * gini_left
                
                # определение прироста информативности очередного правила
                GINI_gain = GINI_base - Pv
                
                # если очередной прирост информативности больше всех
                # предыдущих, то очередное правило записывается, как лучшее,
                # а прирост информативности записывается как максимальный
                if GINI_gain > max_gain:
                    
                    best_feature = feature
                    
                    best_value = value/self.rule_value_step
                    
                    max_gain = GINI_gain
                    
        return(best_feature, best_value)
    
    def grow_tree(self):
        '''
        Метод для выращивания дерева
        '''
        
        df = self.X.copy()
        
        # если не достигнута максимальная глубина дерева и количество элементов
        # в датасете больше минимально допустимого, то продолжается рост
        if((self.depth < self.max_depth) 
           and (self.n >= self.min_samples_split)):
            
            # вычисление правила для наилучшего разделения датасета
            best_feature, best_value = self.best_Split()
            
            # если определён атрибут для лучшего разделения массива, то
            # продолжается рост дерева
            if best_feature is not None:
                
                # разделение датасета
                self.best_feature = best_feature
                self.best_value = best_value
                left_df = df[df[best_feature]<=best_value].copy()
                right_df = df[df[best_feature]>best_value].copy()
                
                # рекурсивное определение левой ветки
                left = Node(left_df,
                            self.label_col,
                            depth = self.depth + 1,
                            max_depth = self.max_depth,
                            rule=f"{best_feature} > {best_value}",
                            node_type = "branch",
                            min_samples_split = self.min_samples_split)
                self.left = left
                
                # рекурсивное выращивание левой ветки
                self.left.grow_tree()
                
                # рекурсивное определение правой ветки
                right = Node(right_df,
                             self.label_col,
                            depth = self.depth + 1,
                            max_depth = self.max_depth,
                            rule = f"{best_feature} > {best_value}",
                            node_type = 'branch',
                            min_samples_split = self.min_samples_split)
                self.right = right 
                
                # рекурсивное выращивание правой ветки
                self.right.grow_tree()
                
            # если не определен лучший атрибут для раздреления, то из узла
            # создаётся лист
            else:
                self.node_type = "Leaf"
        
        # если достигнута максимальная глубина или количество экземпляров
        # во входном датасете меньше минимально допустимого, то из узла
        # создаётся лист
        else:
            self.node_type = "Leaf"
                
    def predict(self, X):
        '''
        Метод для формирования предсказаний для входного датасета на основе
        выращеного дерева

        Parameters
        ----------
        X : pd.DataFrame
            Датасет, для которого формируются предсказания.

        Returns
        -------
        Массив с предсказаниями классов для входного датасета.

        '''
        
        # массив для записи предсказаний
        predictions = []
        
        # цикл для заполнения массива предсказаний
        for i, x in X.iterrows():
            
            # создание словаря значений соотвевтствующих атрибутов для
            # очередного экземпляра
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
                
            # вычисление предсказания с помощью функции прохода по выращенному
            # дереву
            predictions.append(self.prediction_obs(values))
        
        return(predictions)
    
    def prediction_obs(self, values):
        '''
        Метод для определения мажоритарного класса для входного экземпляра
        с помощью прохода по выращенному дереву
        '''
        
        # определение самого объекта в качестве текущего узла
        cur_node = self
        
        # цикл для прохода по дереву
        while(cur_node.depth < cur_node.max_depth):
            
            # в качестве параметров правила разделения выбирается правило
            # разделения текущего объекта
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            
            # если текущий объект является листом, то цикл завершается
            if (cur_node.node_type == "Leaf"):
                break
            
            # если значение атрибута, определённого правилом ветвления
            # текущего объекта, входного экземпляра, меньше значения,
            # определённого этим правилом, то происходит присвоение в
            # качестве текущего объекта левой ветви, иначе - правой
            if(values.get(best_feature) < best_value):
               if self.left is not None:
                   cur_node = cur_node.left
                   
            else:
                if self.right is not None:
                    cur_node = cur_node.right
        
        # из метода возвращается мажоритарный класс полученного в цикле листа
        return cur_node.mag_class
        
    def prun_tree(self, X):
        '''
        Метод для подрезания дерева, на основе dev датасета

        Parameters
        ----------
        X : pd.DataFrame
            development set.

        Returns
        -------
        None.

        '''
        Xsubset = X.copy()
        
        # массив для записи предсказаний
        predictions = []
        
        # массив для прохода по датасету
        for i, x in  Xsubset.iterrows():
            
            # # определение самого объекта в качестве текущего узла
            cur_node = self
            
            # цикл для прохода по выращенному дереву
            while(cur_node.node_type != "Leaf"):
                
                # инкрементирование счётчика использования узла в dev сете
                cur_node.cnt_prun += 1
                
                # в качестве параметров правила разделения выбирается правило
                # разделения текущего объекта
                best_feature = cur_node.best_feature
                best_value = cur_node.best_value
                
                # если мажоритарный класс текущего объекта не совпадает с
                # целевой меткой очередного экземпляра, то инкрементируется
                # счётчик ошибок очередного объекта в качестве листа
                if(cur_node.mag_class != x[self.label_col]):
                    cur_node.errors_as_Leaf += 1
                    
                # если определено правило разделения, то проверяется следующее
                # условие
                if best_feature is not None:
                    # если значение атрибута, определённого правилом ветвления
                    # текущего объекта, входного экземпляра, меньше значения,
                    # определённого этим правилом, то происходит присвоение в
                    # качестве текущего объекта левой ветви, иначе - правой
                    if(x[best_feature] < best_value):
                       if self.left is not None:
                           cur_node = cur_node.left
                           
                    else:
                        if self.right is not None:
                            cur_node = cur_node.right
            
            # после выхода из цикла прохода по дереву инкрементируется счётчик
            # использования итогового листа
            cur_node.cnt_prun += 1
            
            # в качестве предсказания очередного экземпляра определяется
            # мажоритарный класс итогового листа
            predictions.append(cur_node.mag_class)
        
        # удаление неиспользованных веток с помощью соответствующего метода
        self.prun_tree_deleting_zeros()
        
        # счётчик индекса предсказания
        counter = 0
        
        # цикл для прохода по dev сету
        for i, x in Xsubset.iterrows():
            
            # формирование целевой метки очередного экземпляра dev сета
            true_label = x[self.label_col]
            
            # выбор из массива предсказания метки очередного экземпляра dev
            # сета
            pred = predictions[counter]
            
            # инкрементирование счётчика индекса
            counter += 1
            
            # инкрементирование счётчиков ошибок всех узлов дерева с помощью
            # соответствующего метода
            self.prune_tree_add_error_counter(pred, true_label)
        
        # оптимизация листов дерева с помощью соответсвующего метода
        self.prune_tree_delete_leafs()
            
    def prun_tree_deleting_zeros(self):
        '''
        Метод для удаления узлов, которые не были задействованы при
        классификации экземпляров deevelopment сета

        Returns
        -------
        None.

        '''
        
        # Если узел листа не был задействован при классификации экземпляров
        # dev сета, то узел становится листом
        if(self.cnt_prun == 0):
            self.left = None
            self.right = None
            self.node_type = "Leaf"
        else:
            # если узел не является листом, то рекурсивно вызывается текущий
            # метод для правой и левой ветви узла
            if(self.node_type != "Leaf"):
                self.right.prun_tree_deleting_zeros()
                self.left.prun_tree_deleting_zeros()
                

    def prune_tree_add_error_counter(self, pred, true):
        '''
        Метод для инкрементирования счётчика ошибок классификации экземпляров
        development сета

        Parameters
        ----------
        pred : str ot int
            пердсказание класса экземпляра.
        true : str or int
            целевая метка экземпляра.

        Returns
        -------
        None.

        '''
        
        # если пресказание не совпадает с целевой метокой, то счётчик ошибок
        # инкрементируется
        if(true != pred): self.errors += 1
        
        # если узел не является листом, то рекурсивно выывается текущий
        # метод для правой и левой ветви узла
        if(self.node_type != "Leaf"):
            self.right.prune_tree_add_error_counter(pred, true)
            self.left.prune_tree_add_error_counter(pred, true)
                
    def prune_tree_delete_leafs(self):
        '''
        Метод для оптимизации листов дерева
        '''
        
        # если текущий узел не лист
        if(self.left.node_type != "Leaf"):
            # если правая ветвь узла не лист
            if(self.right.node_type != "Leaf"):
                # рекурсивный вызов текущего метода для левого узла
                self.left.prune_tree_delete_leafs()
            else:
                # если ошибок классификации в правой ветви больше, чем ошибок
                # классификации текущего узла, то правый лист удаляется
                # и метод продолжается для левой ветви
                if(self.right.errors >= self.errors_as_Leaf):
                    self = self.left
                    self.prune_tree_delete_leafs()
 
        else:
            # если правая ветвь узла не лист
            if(self.right.node_type != "Leaf"):
                # если ошибок классификации в левой ветви больше, чем ошибок
                # классификации текущего узла, то левый лист удаляется
                # и метод продолжается для правой ветви
                if(self.left.errors >= self.errors_as_Leaf):
                    self = self.right
                    self.prune_tree_delete_leafs()
            # если правая и левая ветви узла являются листами, то происходит
            # оптимизация узла
            else:
                self.make_leaf_for_pruning()
            
    
    def make_leaf_for_pruning(self):
        '''
        Метод для оптимизации узлов с листами
        '''
        
        # выбор минимальной из ошибок классификации узла в качестве листа,
        # суммарных ошибок классификации, ошибок классификации правого листа,
        # ошибок классификации левого листа
        min_error = min(self.errors_as_Leaf, self.errors,
                        self.right.errors, self.left.errors)
        
        # вервление для разных случаев минимальных ошибок
        if (min_error == self.errors_as_Leaf):
            # формирование листа из узла
            self.left = None
            self.right = None
            self.node_type = "Leaf"
        elif(min_error == self.right.errors):
            # переопределение узла правым листом
            self = self.right
        elif(min_error == self.left.errors):
            # переопределение узла левым листом
            self = self.left
        