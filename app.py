
import streamlit as st
import pickle
import pandas as pd

st.write("# Узнай актуальную цену квартиры в НН")
st.write("##### Отказ от ответственности: Пользователь использует приложение на свой страх и риск.")

district = st.selectbox(
    "Район:",
    ['Автозаводский район', 'Другой', 'Канавинский район', 'Ленинский район', 
     'Московский район',  'Нижегородский район', 'Приокский район', 'Советский район', 
     'Сормовский район']
)
st.write("Вы выбрали: ", district)


rooms = ['Комната', '1', '2', '3', '4']
rooms_count = st.selectbox(
    "Количество комнат:",
    rooms,
    index=2
)
st.write("Вы выбрали:", rooms_count)
# теперь создадим словарь, в котором будет обозначение количества комнат и соответсветствующие им числовые значения
room_int = dict(zip(rooms,[0.5,1,2,3,4]))
# и теперь обозначение комнат будет преобразовываться в число


area1 = st.slider(
    '### Общая площадь квартиры (кв.м.):',
    min_value=7.0,
    max_value=200.0,
    value=50.0,
    step=1.0
)
area2 = st.slider(
    '### Жилая площадь квартиры (кв.м.):',
    min_value=7.0,
    max_value=150.0,
    value=50.0,
    step=1.0
)
area3 = st.slider(
    '### Площадь кухни (кв.м.):',
    min_value=0.0,
    max_value=70.0,
    value=9.0,
    step=1.0
)
st.write("Вы указали:", area1, area2, area3)


new_build = st.checkbox("Новостройка:", value=False)

floor = st.number_input(
    "Этаж:", 
    value=5,
    step=1
)

tot_floor = st.number_input(
    "Этажность дома:", 
    value=5,
    step=1
)

st.write("Вы ввели:", floor,'/',tot_floor)



building_type = st.selectbox(
    "Материал дома:",
    ['кирпич', 'панель', 'шлакоблок', 'блок+утеплитель', 'дерево',
           'монолитный железобетон']
)
st.write("Вы выбрали:", building_type)

year = st.number_input(
    "Год постройки", 
    min_value=1822, 
    max_value=2027, 
    value=2000,
    step=1,
    format='%d'
)
st.write("Вы ввели:", year)
# год мы смотрим из discribe. Но у нас год 1822 а максимум 2027. Можно добавить слайдер, но я добавлю просто ввод в поле

# данные мы берем из наших заготовок в датасете. Возникает необходимость где-то хранить эту конфигурацию для наего приложения.
appart =dict(zip(
    ['area1', 'area2', 'area3', 'floor', 'total_floors','district', 'building_type', 'year', 'rooms_count', 'new_build'],
    [area1, area2, area3, floor, tot_floor, district, building_type, year, room_int[rooms_count], new_build]
))

# создадим наш датафрейм и посмотрим, как выглядят наи введенные данные
new = pd.DataFrame(appart, index=[0])
# st.write(new)

# из минусов  в1822 году можем выбрать 100 метровую комнату или 4-х комнатную квартиру в  15 метров. 
# это не очень хорошо, надо иметь возможность отслеживать такие данные и устанавливать границы интервалрв


# ТЕПЕРЬ ПЕРЕХОДИМ К МОДЕЛИ


# ВОЗВРАЩАЕМСЯ
# добавляем чтение модели и получение ответа
model_pkl_file = "data/model/apartment_prices_regression.pkl"
with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)


st.write("## Оценочная стоимость квартиры:","{:,}".format(int(model.predict(new)[0])))











