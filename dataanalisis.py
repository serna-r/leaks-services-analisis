import pandas as pd
import math

i = 0

# Function to analyze password complexity
def pasword_mask(password):
    mask = ''
    if any(c.islower() for c in password): mask = mask + 'l'
    if any(c.isupper() for c in password): mask = mask + 'u'
    if any(c.isnumeric() for c in password): mask = mask + 'd'
    if any(not c.isalnum() for c in password): mask = mask + 's'
    total = True

    # For other languages
    if mask == '': mask = 'z'

    return mask, total

# Calcular entropia formula E = L × log(R) / log(2)
def calcular_entropia(palabra):
    longitud_palabra = len(palabra)
    simbolos_posibles = len(set(palabra))
    
    if simbolos_posibles > 1:  # Para evitar log(1) o log(0)
        entropia = longitud_palabra * (math.log(simbolos_posibles) / math.log(2))
    else:
        entropia = 0  # Si hay un solo símbolo, la entropía es 0
    
    return entropia



def statistics(df, output):
    # Mostrar las primeras filas para asegurarse de que se cargó correctamente
    print(df.head())

    # Generar un informe estadístico de las contraseñas
    # Contar las contraseñas más comunes
    common_passwords = df['Contraseña'].value_counts()

    # print("\nInforme estadístico de contraseñas más comunes:")
    print(common_passwords[:20])

    # Crear una nueva columna 'longitud' que contiene la longitud de cada contraseña
    df['Longitud'] = df['Contraseña'].str.len()

    # Contar la frecuencia de cada longitud de contraseña
    lengthcount = df['Longitud'].value_counts().sort_index()

    # Mostrar el resultado
    print(lengthcount)

    # Apply the function to each password and create a new DataFrame
    df['mask'], df['total'] = zip(*df['Contraseña'].map(pasword_mask))
    # Group by Character Length (for 6, 7, 8, 9, 10) and 'other' for longer or shorter passwords
    bins = [0, 6, 7, 8, 9, 10, float('inf')]
    labels = ['6', '7', '8', '9', '10', 'other']
    df['Length Group'] = pd.cut(df['Longitud'], bins=bins, labels=labels, right=False)
    print(df)

    # Aggregate the data
    table = df.groupby('Length Group', observed=True)['mask'].value_counts().unstack(level=1)
    table['total'] = df.groupby('Length Group', observed=True)['total'].sum()
    table = table.div(table['total'], axis = 0).mul(100, axis=0)
    table = table.reindex(sorted(table.columns), axis=1)
    print(table)

    # Aplicar la función de entropía a cada fila en la columna 'contraseña'
    df['entropia'] = df['Contraseña'].apply(calcular_entropia)

    # Agrupar los valores de entropía en 20 intervalos
    intervalos = pd.cut(df['entropia'], bins=20)

    # Contar cuántas contraseñas hay en cada intervalo
    conteo_intervalos = df.groupby(intervalos)['Contraseña'].count()

    # Mostrar la tabla con los resultados
    print(conteo_intervalos)
  
    # Abrir fichero de output
    f = open(output, 'w')
    # Escribir datos
    f.write(f'Total users read {len(df.index)} \n20 most common passwords \n{common_passwords[:20]} \nLength: \n{lengthcount} \ntable \n{table.to_string()} \n{conteo_intervalos.to_string()}')

if __name__ == '__main__':
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv('datos_extraidos.csv')
    output = 'Stats.txt'

    statistics(df, output)