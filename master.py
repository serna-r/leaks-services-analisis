import os
from dataextract import leer_archivos_en_carpeta
from dataanalisis import statistics

def main():
    # Solicitar el nombre del data leak
    data_leak_name = input("Introduce el nombre del data leak que quieres analizar: ")

    # Definir la ruta del directorio basado en el nombre proporcionado
    data_leak_dir = os.path.join(os.getcwd(), data_leak_name)

    # Verificar si el directorio existe
    if not os.path.isdir(data_leak_dir):
        print(f"El directorio '{data_leak_name}' no existe en la ruta actual.")
        return

    # Ejecutar la extracción de datos
    print("\nEjecutando extracción de datos...")
    data_folder = os.path.join(data_leak_dir, 'data')
    df = leer_archivos_en_carpeta(data_folder)

    if df is None or df.empty:
        print("No se encontraron datos válidos durante la extracción.")
        return

    # Mostrar los datos extraídos (opcional)
    print("\nDatos extraídos:")
    print(df)

    # Ejecutar el análisis de datos
    print("\nEjecutando análisis de datos...")
    output_file = os.path.join(data_leak_dir, 'Stats.txt')
    statistics(df, output_file)

    print(f"\nAnálisis completado con éxito. Revisa el archivo '{output_file}' para los resultados.")

if __name__ == "__main__":
    main()