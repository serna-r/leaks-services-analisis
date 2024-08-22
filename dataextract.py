import os
import pandas as pd
import re

verbose = 0
def leer_archivos_en_carpeta(carpeta):
    archivos = os.listdir(carpeta)
    datos = []

    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta, archivo)

        if verbose < 0: print('Abriendo archivo: ', ruta_archivo)

        with open(ruta_archivo, 'r', encoding='utf-8') as file:
            for linea in file:
                correo, contraseña = separar_correo_y_contrasena(linea)
                if correo is not None and contraseña is not None:
                    # Check also for unknoun or null passwords
                    if not (contraseña == 'NULL' or contraseña == 'none' or contraseña == '?'
                            or contraseña == 'None'):
                        datos.append([correo, contraseña])
                    if verbose > 1: print(f"correo: {correo} contraseña: {contraseña}")

    # Crear un DataFrame de pandas con todos los datos recopilados
    df = pd.DataFrame(datos, columns=['Usuario', 'Contraseña'])
    return df

def leerunarchivo(carpeta, numero_archivo):
    # Obtener la lista de archivos en la carpeta
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.txt')]

    # Verificar si el número de archivo es válido
    if numero_archivo < 1 or numero_archivo > len(archivos):
        print("Número de archivo no válido. Por favor, elige un número entre 1 y", len(archivos))
        return None


    # Leer el archivo seleccionado
    archivo_seleccionado = archivos[numero_archivo - 1]
    ruta_archivo = os.path.join(carpeta, archivo_seleccionado)

    datos = []

    if verbose > 0: print('Abriendo archivo: ', ruta_archivo)

    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        for linea in file:
            correo, contraseña = separar_correo_y_contrasena(linea)
            if correo is not None and contraseña is not None:
                datos.append([correo, contraseña])
                if verbose > 1: print(f"correo: {correo} contraseña: {contraseña}")
        # Crear un DataFrame de pandas
        df = pd.DataFrame(datos, columns=['Usuario', 'Contraseña'])
        return df
    
def separar_correo_y_contrasena(cadena):
    # Expresión regular para capturar correo y contraseña
    patron = r"([^:]+)@([^:]+):(.+)"
    
    # Buscar coincidencias
    coincidencia = re.match(patron, cadena)
    
    if coincidencia:
        correo = coincidencia.group(1) + "@" + coincidencia.group(2)
        contrasena = coincidencia.group(3)
        return correo, contrasena
    else:
        return None, None


def extract():
    carpeta = input("Introduce el nombre de la carpeta")

    df = leer_archivos_en_carpeta(carpeta + '/data')

    if df is not None:
        print("\nDatos leídos:")
        print(df)
        
        df.to_csv(carpeta + '/datos_extraidos.csv', index=False)


if __name__ == '__main__':
    extract()
