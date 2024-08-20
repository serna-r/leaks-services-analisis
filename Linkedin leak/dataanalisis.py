import pandas as pd

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
    table.to_csv('stats.csv', sep=";", decimal=",")
    
  
    # Abrir fichero de output
    f = open(output, 'w')
    # Escribir datos
    f.write(f'Total users read {len(df.index)} \n20 most common passwords \n{common_passwords[:20]} \nLength: \n{lengthcount} \ntable \n{table.to_string()}')

if __name__ == '__main__':
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv('datos_extraidos.csv')
    output = 'Stats.txt'

    statistics(df, output)