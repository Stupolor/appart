import pandas as pd

data = pd.read_csv('data/raw/nnapartment_more_info.csv')
pd.set_option('display.width', None)

data.drop_duplicates(inplace=True)

grouped_data = data.groupby(['building_type'])
data['year'] = grouped_data['year'].transform(lambda x : x.fillna(x.mean()))
data['year'] = data.year.astype('int')

cols = ['rooms', 'area1', 'area2', 'area3', 'district', 'floor', 'total_floors',
       'building_type', 'year', 'price']
data = data[cols].copy()

data['rooms_count'] = data.rooms.replace('К','0.5').astype('float')

data.drop(['rooms'], axis=1, inplace=True)

dist = [x for x in data.district.unique() if 'район' in x]

data.loc[~data.district.isin(dist), 'district'] = 'Другой'
data["new_build"] = (data.year >= 2024).astype(int)
data = data[data['year'] >= 1922]

data.to_csv('data/processed/nnapartment_more_info.csv', index=False)