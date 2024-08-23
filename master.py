import os
import time  # Import the time module
from dataextract import read_files_in_folder
from dataanalisis import statistics

verbose = 0

def main():
    # Read the leak names from a text file
    with open("leaks.txt", "r") as file:
        leak_names = file.read().splitlines()

    for data_leak_name in leak_names:
        if not data_leak_name.strip():
            continue  # Skip empty lines

        print(f"\nProcessing data leak: {data_leak_name}")

        # Define the directory path based on the provided name
        data_leak_dir = os.path.join(os.getcwd(), data_leak_name)

        # Check if the directory exists
        if not os.path.isdir(data_leak_dir):
            print(f"The directory '{data_leak_name}' does not exist in the current path.")
            continue

        # Execute data extraction
        print("\nExecuting data extraction...")
        start_extract_time = time.time()  # Start time of extraction
        data_folder = os.path.join(data_leak_dir, 'data')
        df = read_files_in_folder(data_folder)
        end_extract_time = time.time()  # End time of extraction

        if df is None or df.empty:
            print("No valid data found during extraction.")
            continue

        # Calculate elapsed time for extraction
        time_elapsed_extract = end_extract_time - start_extract_time
        print(f"Elapsed time for data extraction: {time_elapsed_extract:.2f} seconds.")

        # Optionally display the extracted data
        if verbose >= 0:
            print("\nExtracted data:", len(df.index), ' passwords')
        if verbose > 0:
            print(df)

        # Execute data analysis
        print("\nExecuting data analysis...")
        start_analysis_time = time.time()  # Start time of analysis
        output_file = os.path.join(data_leak_dir, 'Stats.txt')
        statistics(df, output_file)
        end_analysis_time = time.time()  # End time of analysis

        # Calculate elapsed time for analysis
        time_elapsed_analysis = end_analysis_time - start_analysis_time
        print(f"Elapsed time for data analysis: {time_elapsed_analysis:.2f} seconds.")

        print(f"\nAnalysis completed successfully. Check the file '{output_file}' for results.")

if __name__ == "__main__":
    main()