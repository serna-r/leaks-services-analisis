import os
import sys
import time
import traceback
from dataextract import read_files_in_folder
from dataanalisis import statistics
from dataanalisis import one_stat
from write_latex import get_latex_table
from clustering import clustering
from distribution_comparison import get_distribution_comparison
from servicesanalisis import service_analisis
from datetime import datetime

verbose = 0
# File for leak analysis
leak_stats_file = "leaks.txt"
# File for leak distributions
leak_distribution_file='leak_types.txt'
# Special extraction modes
modes = ['extract_hex', 'only_password']
# Single stat
stats = ['simple_entropy', 'shannon_entropy', 'password_strength', 'password_score_and_length']

def print_and_log(message, log_file=None):
    with open(log_file, 'a') as log_file_descriptor:
        # Prints a message to both the console and a log file.
        print(message)  # Print to console
        if log_file_descriptor:
            print('[' + str(datetime.now()) + ']: ' + message, file=log_file_descriptor)  # Print to file

def process_leaks():
    total_start_time = time.time()  # Start time for the entire process

    # Read the leak names from a text file
    with open(leak_stats_file, "r") as file:
        leak_names = file.read().splitlines()

    # Log file
    log_file = 'log.txt'

    # For each leak call function to process it
    for data_leak_name in leak_names:
        # Handle errors for each leak
        try:
            process_leak(data_leak_name, log_file)
        except Exception as e:
            # Log the exception and continue with the next leak
            print_and_log(f"Error processing data leak {data_leak_name}: {e}", log_file)
            print_and_log(f"Error trace:\n {traceback.format_exc()}", log_file)
            continue  # Skip to the next leak

    # Calculate the total elapsed time for all leaks
    total_end_time = time.time()
    total_time_elapsed = total_end_time - total_start_time
    print_and_log(f"\n\nTotal time elapsed for all data leaks: {total_time_elapsed:.2f} seconds.\n", log_file)

def process_leak(data_leak_name, log_file):
    # Calculate leak start time
    leak_start_time = time.time()

    if not data_leak_name.strip():
        return  # Skip empty lines
    elif data_leak_name.startswith('#'):
        print_and_log(f"Skip {data_leak_name.replace('#', '')}", log_file)
        return # Skip commented lines
    
    data_leak_name = data_leak_name.strip()
    # Mode selection
    mode = None
    if data_leak_name.split()[-1] in modes:
        mode = data_leak_name.split()[-1]
        data_leak_name = data_leak_name.rsplit(' ', 1)[0].strip()

    stat = 'all'
    # Stat selection
    if data_leak_name.split()[-1] in stats:
        stat = data_leak_name.split()[-1]
        data_leak_name = data_leak_name.rsplit(' ', 1)[0].strip()

    print_and_log(f"Processing data leak: {data_leak_name}, mode: {mode}", log_file)

    # Define the directory path based on the provided name
    data_leak_dir = os.path.join(os.getcwd(), 'leaks/' + data_leak_name)

    # Check if the directory exists
    if not os.path.isdir(data_leak_dir):
        print_and_log(f"The directory '{data_leak_name}' does not exist in the current path.", log_file)
        return
    # Execute data extraction
    print_and_log("Executing data extraction...", log_file)
    start_extract_time = time.time()  # Start time of extraction
    data_folder = os.path.join(data_leak_dir, 'data')
    df = read_files_in_folder(data_folder, mode)
    end_extract_time = time.time()  # End time of extraction

    if df is None or len(df.index) == 0:
        print_and_log("No valid data found during extraction.", log_file)
        return

    # Calculate elapsed time for extraction
    time_elapsed_extract = end_extract_time - start_extract_time
    print_and_log(f"Elapsed time for data extraction: {time_elapsed_extract:.2f} seconds.", log_file)

    # Optionally display the extracted data
    if verbose >= 0:
        print_and_log(f"Extracted data: {len(df.index)} passwords", log_file)
    if verbose > 0:
        print_and_log(str(df), log_file)

    # Execute data analysis
    if verbose >= 0:
        print_and_log("Executing data analysis...", log_file)
    start_analysis_time = time.time()  # Start time of analysis

    # Check for single stat or all stats
    if stat in stats:
        output_file = os.path.join(data_leak_dir, stat + '.txt')
        print_and_log("Single stat:" + stat, log_file)
        one_stat(df, stat, output_file)    
    else:
        output_file = os.path.join(data_leak_dir, 'Stats.txt')
        print_and_log("All stats", log_file)
        statistics(df, output_file)
    
    end_analysis_time = time.time()  # End time of analysis

    # Calculate elapsed time for analysis
    time_elapsed_analysis = end_analysis_time - start_analysis_time
    print_and_log(f"Elapsed time for data analysis: {time_elapsed_analysis:.2f} seconds.", log_file)

    print_and_log(f"Analysis completed successfully. Check the file '{output_file}' for results.", log_file)

    # Calculate leak time
    leak_end_time = time.time()
    leak_time_elapsed = leak_end_time - leak_start_time
    print_and_log(f"\nTotal time elapsed for data leak: {leak_time_elapsed:.2f} seconds.\n", log_file)


def get_distribution_comparison_with_logging():
    log_file = 'log.txt'

    # Start time for the distribution comparison process
    start_time = time.time()
    print_and_log("Starting distribution comparison...", log_file)

    # Call the actual distribution comparison function
    get_distribution_comparison(leak_distribution_file)

    # End time for the process
    end_time = time.time()
    elapsed_time = end_time - start_time

    print_and_log(f"Distribution comparison completed successfully.", log_file)
    print_and_log("Graphs saved in figures folder", log_file)
    print_and_log(f"Elapsed time for distribution comparison: {elapsed_time:.2f} seconds.\n", log_file)

def get_latex_with_logging():
    log_file = 'log.txt'

    # Start time for the latex file generation
    start_time = time.time()
    print_and_log("Starting latex file generation ...", log_file)

    # Generate the LaTeX table
    output_path = 'latex/distribution_resume_table.tex'
    get_latex_table(leak_distribution_file ,output_path)

    # End time for the process
    end_time = time.time()
    elapsed_time = end_time - start_time

    print_and_log(f"Latex file generation completed successfully.", log_file)
    print_and_log("File saved in latex folder", log_file)
    print_and_log(f"Elapsed time for latex file generation: {elapsed_time:.2f} seconds.\n", log_file)

def get_cluster_with_logging(kmeansTest=False):
    log_file = 'log.txt'

    # Start time for the latex file generation
    start_time = time.time()
    print_and_log("Starting clustering ...", log_file)

    # Call cluster function
    leaks_file='leak_types.txt'
    clustering(leaks_file, kmeans=kmeansTest)

    # End time for the process
    end_time = time.time()
    elapsed_time = end_time - start_time

    print_and_log(f"Clustering completed successfully.", log_file)
    print_and_log("Graphs saved in figures folder", log_file)
    print_and_log(f"Elapsed time for clustering: {elapsed_time:.2f} seconds.\n", log_file)

def get_serviceanalisis_with_logging():
    log_file = 'log.txt'

    # Start time for the latex file generation
    start_time = time.time()
    print_and_log("Starting service data analisis ...", log_file)

    # Call service data analisis
    service_data ='./services/services.xlsx'
    service_analisis(service_data)

    # End time for the process
    end_time = time.time()
    elapsed_time = end_time - start_time

    print_and_log(f"Service data analisis completed successfully.", log_file)
    print_and_log("Graphs saved in figures folder", log_file)
    print_and_log(f"Elapsed time for service data analisis: {elapsed_time:.2f} seconds.\n", log_file)


def print_help():
    print("Usage: python master.py [option]")
    print("Options:")
    print("-s --stats: Process leaks and gather statistics. (file leaks.txt)")
    print("-d --distributioncomparison: Get distribution comparison. (file leak_types.txt)")
    print("-l --latex: Make a latex file in the latex folder with important data (file leak_types.txt)")
    print("-c --cluster: Execute the cluster module")
    print("-sa --serviceanalisis: execute service data analisis from the data colected in the excel document (./services/services.xlsx)")
    print("-h --help: Display this help menu")
    print("No option: Display this help menu")

def main():
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print_help()
    elif len(sys.argv) >= 2:
        option = sys.argv[1]
        
        if option in ['-s', '--stats']:
            process_leaks()
        elif option in ['-d', '--distributioncomparison']:
            get_distribution_comparison_with_logging()
        elif option in ['-l', '--latex']:
            get_latex_with_logging()
        elif option in ['-c', '--cluster']:
            if len(sys.argv) == 3 and sys.argv[2] == 'Kmeans':
                get_cluster_with_logging(kmeansTest=True)
            else:
               get_cluster_with_logging() 
        elif option in ['-sa', '--serviceanalisis']:
            get_serviceanalisis_with_logging()
        elif option in ['-h', '--help']:
            print_help()
        else:
            print("Invalid option. Please try again.")
            print_help()
    else:
        print("Invalid usage. Please use only one argument.")
        print_help()

if __name__ == "__main__":
    main()
