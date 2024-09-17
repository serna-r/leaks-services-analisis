import os
import time  # Import the time module
from dataextract import read_files_in_folder
from dataanalisis import statistics
from dataanalisis import one_stat
from datetime import datetime

# Assuming get_distribution_comparison is imported
from distribution_comparison import get_distribution_comparison

verbose = 0
# File for leak analysis
leak_stats_file = "leaks.txt"
# File for leak distributions
leak_distribution_file='leak_types.txt'
# Special extraction modes
modes = ['extract_hex']
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

    for data_leak_name in leak_names:
        # Calculate leak start time
        leak_start_time = time.time()
        # Log file
        log_file = 'log.txt'

        if not data_leak_name.strip():
            continue  # Skip empty lines
        elif data_leak_name.startswith('#'):
            print_and_log(f"Skip {data_leak_name.replace('#', '')}", log_file)
            continue # Skip commented lines
        
        # Mode selection
        mode = None
        if data_leak_name.split()[-1] in modes:
            mode = data_leak_name.split()[-1]
            data_leak_name = data_leak_name.rsplit(' ', 1)[0]

        stat = 'all'
        # Stat selection
        if data_leak_name.split()[-1] in stats:
            stat = data_leak_name.split()[-1]
            data_leak_name = data_leak_name.rsplit(' ', 1)[0]

        print_and_log(f"Processing data leak: {data_leak_name}, mode: {mode}", log_file)

        # Define the directory path based on the provided name
        data_leak_dir = os.path.join(os.getcwd(), data_leak_name)

        # Check if the directory exists
        if not os.path.isdir(data_leak_dir):
            print_and_log(f"The directory '{data_leak_name}' does not exist in the current path.", log_file)
            continue

        # Execute data extraction
        print_and_log("Executing data extraction...", log_file)
        start_extract_time = time.time()  # Start time of extraction
        data_folder = os.path.join(data_leak_dir, 'data')
        df = read_files_in_folder(data_folder, mode)
        end_extract_time = time.time()  # End time of extraction

        if df is None or len(df.index) == 0:
            print_and_log("No valid data found during extraction.", log_file)
            continue

        # Calculate elapsed time for extraction
        time_elapsed_extract = end_extract_time - start_extract_time
        print_and_log(f"Elapsed time for data extraction: {time_elapsed_extract:.2f} seconds.", log_file)

        # Optionally display the extracted data
        if verbose >= 0:
            print_and_log(f"Extracted data: {len(df.index)} passwords", log_file)
        if verbose > 0:
            print_and_log(str(df), log_file)

        # Execute data analysis
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

    # Calculate the total elapsed time for all leaks
    total_end_time = time.time()
    total_time_elapsed = total_end_time - total_start_time
    print_and_log(f"\n\nTotal time elapsed for all data leaks: {total_time_elapsed:.2f} seconds.\n", log_file)

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
    print_and_log(f"Elapsed time for distribution comparison: {elapsed_time:.2f} seconds.\n", log_file)


def menu():
    while True:
        print("\nMenu:")
        print("1. Process leaks and gather statistics. (file leaks.txt)")
        print("2. Get distribution comparison. (file leak_types.txt)")
        print("3. Exit.")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            process_leaks()
        elif choice == '2':
            get_distribution_comparison_with_logging()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
