import numpy as np
import pandas as pd
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

# creamos la serie, pasando una lista de valor y. panda crea
# un integer index por defecto
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# creamos in dataframe pasandole a NumPy un arreglo, con una fecha
# de entrada y el numero de columbas
dates = pd.date_range('20190101', periods=6)
print(dates)

# df = pd.dataframe()
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

# creamos un DataFrame con un diccionario de objetos que pueden ser convertidos
# en una serie

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20190102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

print(df2)

# las columnas del resultado DataFrame tienen diferentes dtypes

print(df2.dtypes)

#print(df2.<TAB>)  # noqa: E225, E999

# vemos el top y botton del frame
print(df.head())
print(df.tail(3))

# mostramos el index, columnas
print('index/columnas =')
print(df.index)
print(df.columns)

print(df.to_numpy())
print(df2.to_numpy())

# mostramos un resumen statico de nuestra Data
print(df.describe())

# transponemos nuestra Data
print(df.T)

# sorting by an axis
print(df.sort_index(axis=1, ascending=False))
# sorting by values
print(df.sort_values(by='B'))

# seleccionamos una sola columna
print(df['A'])

# seleccion por filas
print('seleccion filas =')
print(df[0:3])
print(df['20190102':'20190104'])

# mas selecciones por label
print('seleccion label =')
print(df.loc[dates[0]])
print(df.loc[:, ['A', 'B']])
print(df.loc['20190102':'20190104', ['A', 'B']])
print(df.loc[dates[0], 'A'])
print(df.at[dates[0], 'A'])

# seleccion por posicion
print('seleccion posicion =')
print(df.iloc[3])
print(df.iloc[3:5, 0:2])
print(df.iloc[[1, 2, 4], [0, 2]])
print(df.iloc[1:3, :])
print(df.iloc[:, 1:3])
print(df.iloc[1, 1])
print(df.iat[1, 1])

# filtramos data por valores
print('boolean_inexing =')
print(df[df.A > 0])
print(df[df > 0])

# isin() method for filtering
# agregamos una columna
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print('filtrando =')
print(df2)
# filtramos
print(df2[df2['E'].isin(['two', 'four'])])

####Settings#####
# nueva columna automaticamente alineada con la informacion por el index

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20190102', periods=6))
print(s1)

df['F'] = s1

# setting value by llabel
df.at[dates[0], 'A'] = 0
# setting value by position
df.iat[0, 1] = 0
# setting by assigning wit NumPy array
df.loc[:, 'D'] = np.array([5] * len(df))

print(df)

# operation with setting
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

#####Missing Data#####
print('missind data =')
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
print(df1)

# to drop any rows that have miissinng data
print(df1.dropna(how='any'))
# filling missing data
print(df1.fillna(value=5))
# to get the boolean mask where values are Nan
print(pd.isna(df1))

# Operation##3
# operations in general eclude missing data

print('operation =')
print(df.mean())
# operations with objects taht have different
# dimensionality and need alignment
s = pd.Series([1, 2, 5, np.nan, 6, 8], index=dates).shift(2)
print(s)

print(df.sub(s, axis='index'))

# apply function to the data
print(df.apply(np.cumsum))
print(df.apply(lambda x: x.max() - x.min()))

# histograming###3
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())

##Strin Methods
s = pd.Series(['A','B','C','Aaba','Bacca',np.nan,'CABA','dog','cat'])
print(s.str.lower())

