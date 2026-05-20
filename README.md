# Однослойный перцептрон
## Гончарова Виктория Б25-527 
Реализован алгоритм обучения однослойного перцептрона без готовых библиотек глубокого обучения (PyTorch, TensorFlow, Keras, sklearn.linear_model). Проведены исследования влияния параметров, различных функций потерь на процесс обучения, показано, как меняется эффективность модели при разном распределнии данных в двумерном пространстве, изучены различные метрики качества работы модели. 

Код подразделён на 3 модуля: main.py для запуска обучения и экспериментов, perceptron.py, содержащий класс, реалищующий модель, и utils.py, содержащий вспомогательные функции для проведения экспериментов и отображения графиков.

### perceptron.py

Содержит класс SingleLayerPerceptron, имеющий следующие функции:
- init: инициализация модели кол-вом признаков данных, масштабом значений весов, типом функции потерь и значением learning rate;
- sigmoid: функция активации для приведения предсказанных данных в вероятностные значения по формуле: $$\sigma(z) = \frac{1}{1 + e^{-z}}$$, имеется защита от слишком больших значений экспоненты;
- forward: прямой ход модели, предсказание вероятностных значений с помощью формулы $$z = w^T X + b = \sum_{i=1}^{n} w_i x_i + b$$ и применения функции активации;
- compute_loss: значение функции потерь, заданной пользователем:
  
  *бинарная кросс-энтропия*: $$L_{BCE}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$ - классическая функция потерь для задачи классификации
  
  или *hinge loss*: $$L_{hinge}(y, z) = \max\left(0, 1 - y \cdot z\right)$$; - функция потерь, использующаяся в методе SVM, максимизирующая зазор между классами.

  Применяется также l2-регулязация: $$R_{L2}(w) = \frac{\lambda}{2} \sum_{j=1}^{n} w_j^2$$;
- fit: обновление весов модели за заданное количество эпох, на батчах заданного размера. Используется Стохастический градиентный спуск SGD:

  $$w^{(t+1)} = w^{(t)} - \eta \cdot \frac{\partial L}{\partial w}$$

  $$b^{(t+1)} = b^{(t)} - \eta \cdot \frac{\partial L}{\partial b}$$, где η - learning rate
  
  $$\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x{ij}$$

  или при l2-регулязации:  $$\frac{\partial L_{total}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x{ij} + \lambda w_j$$ - модель "штрафуется" за большие веса для предотвращения переобучения
  
  $$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

  При задании $$\beta$$ применяется SGD Momentum для накопления "скорости" для ускорения сходимости и подавления осцилляций:

  $$v_w^{(t)} = \beta \cdot v_w^{(t-1)} + \eta \cdot \frac{\partial L}{\partial w}$$

  $$v_b^{(t)} = \beta \cdot v_b^{(t-1)} + \eta \cdot \frac{\partial L}{\partial b}$$

  $$w^{(t+1)} = w^{(t)} - v_w^{(t)}$$

  $$b^{(t+1)} = b^{(t)} - v_b^{(t)}$$;
- predict: формируется результирующий вектор предсказаний классов в зависимости от типа функции потерь.

### utils.py

- loss_graph и dividing_line_graph: функции для отрисовки с помощью matplotlib.pyplot графика значении лосса в течении каждой эпохи обучени и графика, визуализирующего разделяющую границу на фоне точек данных;
- n_-, batch_-, w_experiment: функции для проведения экспериментов с различными значениями параметров модели;
- generate_synthetic_data: функция для генерации синтетических наборов данных разного вида: линейно разделимые данные - два гауссовых облака, точки на окружности и внутри и снаружи неё и XOR - точки расположены по углам квадрата.

### Результаты обучения и экспериментов

После первого обучения мы получили следующие графики лосса и разделяющей прямой:
<img width="2511" height="1158" alt="image" src="https://github.com/user-attachments/assets/a3bb029c-f32e-46db-b6bb-3d83a89337e7" />

- Эксперимент с learning rate (0.001, 0.01, 0.5, 1):
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/5133172a-35c8-4d18-b6ca-f9bf6313ee3c" /> 
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/c672d536-57f0-4a9f-a985-f1b77190f06b" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/f3a8f5cc-b0c0-4b54-8ad7-3e7532724218" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/dc1d6385-934e-4b51-b0f3-06d6cda7e40d" />
<img width="1086" height="192" alt="image" src="https://github.com/user-attachments/assets/03e9bc13-f29d-4b16-9a7e-da7d438d7d1c" />

- Эксперимент с размером батча (1, 16, 64, 256):
<img width="1000" height="1000" alt="batch_size_experiment_1" src="https://github.com/user-attachments/assets/0ca222a2-4021-4176-a9ca-e433f932091b" />
<img width="1000" height="1000" alt="batch_size_experiment_2" src="https://github.com/user-attachments/assets/5f986241-7f79-4b39-b09e-84fbcc028399" />
<img width="1000" height="1000" alt="batch_size_experiment_3" src="https://github.com/user-attachments/assets/f5b22626-e929-4dd5-9bbe-ac51e3699cdd" />
<img width="1000" height="1000" alt="batch_size_experiment_4" src="https://github.com/user-attachments/assets/63eba5b4-de26-4ef4-8360-24aacff25095" />
<img width="1084" height="191" alt="image" src="https://github.com/user-attachments/assets/9718fbb9-12d5-41c0-b225-87a67c0e85c5" />

- Эксперимент с инициализацией весов (нулевые, маленькие, от 1 до 10):
<img width="1000" height="1000" alt="weight_init_experiment_1" src="https://github.com/user-attachments/assets/8cce0f47-0725-4879-987b-484ee38db1e7" />
<img width="1000" height="1000" alt="weight_init_experiment_2" src="https://github.com/user-attachments/assets/f5901b5a-5f42-4da6-a67e-c19e52f986d7" />
<img width="1000" height="1000" alt="weight_init_experiment_3" src="https://github.com/user-attachments/assets/8b045cc7-1988-4efc-9325-858a5808f9a7" />
<img width="1110" height="258" alt="image" src="https://github.com/user-attachments/assets/d75c9b43-d4a8-4d41-8b81-02a52f914d1d" />

- Эксперимент с различным распределением данных в пространстве:
<img width="2321" height="1118" alt="image" src="https://github.com/user-attachments/assets/8c1f8063-6954-4aad-b96d-8b0781b1a703" />
<img width="2341" height="1134" alt="image" src="https://github.com/user-attachments/assets/9cd9527e-0ae0-4cff-94c5-84c0b9eced64" />
<img width="2338" height="1125" alt="image" src="https://github.com/user-attachments/assets/76defce5-72b0-4b6d-aa94-0024de101c2f" />

Мы можем сделать вывод о том, что однослойный перцептрон -- подходящая модель только для линейно разделимых данных.

- Сравнение BCE и Hinge loss:
<img width="2343" height="1131" alt="image" src="https://github.com/user-attachments/assets/1aa6474c-6f12-41b2-a1ac-fc2eead7a3db" />

- Эксперимент с l2-регулязацией (0,0.001, 0.1, 2, 10):
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/b95744ce-34a5-4b11-8cca-4c2fb5ebead8" />
<img width="1000" height="1000" alt="loss_with_l2_0 001" src="https://github.com/user-attachments/assets/ef566afd-1409-41ee-879a-db6a9641e73f" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/3573b9d9-eb8a-4e2f-94f6-25c1d18f5f9c" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/7adb2bb5-0337-49e7-b9ef-a0b07304a2f3" />
<img width="1300" height="1300" alt="image" src="https://github.com/user-attachments/assets/b0b18f28-8826-47c3-9d13-51dfad9aaa0f" />

- Различные метрики, ROC-прямая и неправильно вычисленные значения:
<img width="349" height="168" alt="image" src="https://github.com/user-attachments/assets/70e6c87c-52b7-46b7-bfe8-2a78273b7548" />
<img width="2310" height="1111" alt="image" src="https://github.com/user-attachments/assets/1ddbbc42-5a1f-45fd-a5d4-ff00cbc5962c" />

- Исследование сходимости градиентного спуска (beta = 0, 0.5, 0.9, 0.99):
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/19907c21-a04c-4b37-b41f-a80aafe2b12d" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/b32a94fe-9ba1-4dae-a868-bd586db15368" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/3e63bb04-752f-4dcc-adec-bf6f0dc69127" />
<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/8d70f314-0909-4eaa-8591-477214882424" />

- Пятикратная кросс-валидация, подбор лучших гиперпараметров и финальная модель
<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/19828b8c-d3e8-41d3-9107-c2285f6f5a33" />
<img width="2308" height="1115" alt="image" src="https://github.com/user-attachments/assets/3d0af6ab-6053-4a20-a6ac-017b65bbb8cd" />

