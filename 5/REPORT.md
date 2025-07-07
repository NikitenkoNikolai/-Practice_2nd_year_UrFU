# Задание 1: Стандартные аугментации torchvision
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/1.1.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/1.2.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/1.3.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/1.4.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/1.5.JPG)
# Итог по заданию 1
В общем, удалось применить для 5 различных классов применить 6 аугментаций по отдельности, а также применить их всех в 1-одном изображении 

# Задание 2: Кастомные аугментации
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/2.1.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/2.2.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/2.3.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/2.4.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/2.5.JPG)
# Итог по заданию 2
Мне удалось создать свои аугментации и комбинировать их, а сравнивать с ранее испльзованными аугментациями не вижу смысла, что сравнивать, что изображения разные 
после эффектов своих аугментаций ? По времени тоже особой разници не заметил.  

# Задание 3: Анализ датасета 
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/3.1.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/3.2.JPG)
# Итог задание 3
По распределению количества изображения у каждого класса можно понять, что абсолютно все классы имеют одинаковое количестов изображений.
А вот по распределению размеров изображения все интереснее. Можно заметить, что большинство изображений имеют одинаковую ширину (это где-то 550-570px), а высота почти 
у каждой картинки максимально разные (есть конечно совпадения, но их количество мало) 

# Задание 4: Pipeline аугментаций
вывод программы
Применение конфигурации: light
Аугментации: ['ToTensor', 'RandomHorizontalFlip', 'ColorJitter']
Применение конфигурации: medium
Аугментации: ['ToTensor', 'RandomHorizontalFlip', 'RandomCrop', 'ColorJitter', 'RandomGrayscale']
Применение конфигурации: heavy
Аугментации: ['ToTensor', 'RandomHorizontalFlip', 'RandomCrop', 'ColorJitter', 'RandomGrayscale', 'AddGaussianNoise', 'CutOut', 'ElasticTransform']
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/augment_exp.JPG)
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/augment.JPG)
# Итог задания 4
Особо сказать нечего. 
Вывод программы показывает, какие наборы аугментаций применяются в зависимости от уровня сложности (light, medium, heavy).
Это позволяет понять, насколько интенсивно изменяются изображения перед обучением модели.
Light — минимальные изменения: отражение и небольшие цветовые вариации. Подходит для базового расширения датасета.
Medium — добавляется обрезка и ЧБ преобразование. Улучшает устойчивость к разным условиям.
Heavy — самые сильные аугментации: шумы, вырезание участков, деформации. Полезно при малом количестве данных.

# Задание 5: Эксперимент с размерами
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/5.1.JPG)
# Итог задания 5
В общем можно было ожидать, что от увеличения мастштаба изображения будет увеличиваться и размер использованной памяти и время выполнения. 

# Задание 6: Дообучение предобученных моделей 
![Image alt](https://github.com/NikitenkoNikolai/-Practice_2nd_year_UrFU/tree/main/5/imgs/6.1.JPG)
# Итог задания 6
Судя по графикам модель, которую я выбрал (resnet18), показывает хорошие более результаты. Однако есть беспокойство, например на эпохе 4, и особенно на 6 можно заметить гигантские скачки,
что ухудшают результаты. Главное то, что к концу обучения все показатели становятся лучше, даже слишком, так как loss и accuracy тестового этапа становится лучше тренировачных, что может говорить 
о возможном переобучении. К счастью вроде в 10-ой эпохе стало все приходить в норму 
