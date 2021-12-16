import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from numpy import array
from random import seed, randint

# функция генерации данных
def create_data(count, high):

    data = list()
    sums = list()

    for n in range(count):
        # генерация пары чисел для сложения
        data.append([randint(0, high), randint(0, high)])
        # подсчет сумм этих пар чисел
        sums.append(sum(data[n]))

    data = array(data)
    sums = array(sums)
    return data, sums

# функция настройки модели
def setup(model):
    model.add(Flatten(input_dim=2)) # выравнивание входного массива в вектор
    # добавление двух слоев с 20 узлами в каждом
    # relu - функция активации (выпрямленная линейная единица)
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    # модель выдает единственное выходное значение суммы (регрессионная модель)
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics='mae')

# функция обучения модели
# при каждом повторении создаются новые данные,
# которые подаются в сеть по 10 раз
def train(model):
    for i in range(15): # количество повторений
        X, y = create_data(pairs, maxValue)
        model.fit(X, y, epochs=10, batch_size=10, shuffle=True)

# функцмя сохранения модели
# позволяет загружать уже обученную модель
def save_model(model, filename):
    model.save(filename)

#-------------------------------------------------------------------------
seed(100)
pairs = 100 # количество пар чисел, генерируемых для обучения модели
maxValue = 100 # максимально допустимое значение числа
model = Sequential() # создание экземпляра модели

# выбор, использовать существующую модель или создать новую
use_existing_model = False

if use_existing_model:
    model = load_model('lab3model.h5')
else:
    setup(model)
    train(model)
    model.save('lab3model.h5')

# проверка обученной модели на тестовом наборе данных
X, y = create_data(20, maxValue)
testresult = model.predict(X, batch_size=1)

# оценивание обученной модели
test_loss, test_acc = model.evaluate(X, y)
print('Test mean square error: ', test_loss) # вывод среднеквадратичной ошибки

# вывод результата проверки модели
for i in range(len(testresult)):
    print('{:4d} {:12.6f} {:12.6f} {:12.6f} {:4d}'.format(i+1, X[i][0], X[i][1], testresult[i][0], y[i]))
