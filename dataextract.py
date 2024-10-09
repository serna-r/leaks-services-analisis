import os
import pandas as pd
import re
import modes

verbose = 0

def read_files_in_folder(folder, mode=None):
    files = os.listdir(folder)
    data = []

    for file in files:
        file_path = os.path.join(folder, file)

        if verbose < 0: print('Opening file: ', file_path)

        # Open file
        with open(file_path, 'r', encoding='utf-8') as f:
            # If the mode is only password there is a password per line
            if mode == 'only_password': data = [line.strip() for line in f if line.strip()]
            # Else separate
            else:
                try:
                    for line in f:
                        # Split email and pass
                        email, password = split_email_and_password(line)
                        # If there is a mode selected apply it
                        if mode != None: password = mode_select(password, line, mode)
                        # Check for missing password
                        if password is not None:
                            # Check also for unknown or null passwords
                            if password == 'NULL' or password == 'none' or password == '?' or password == 'None' or password == "unknown" or password == '(null)':
                                continue

                            data.append([password])
                            if verbose > 1: print(f"email: {email} password: {password}")
                except Exception as e:
                    print(f"Error in line:\n {line} \n Exception:\n {e}\n")

    # Create a pandas DataFrame with all the collected data
    df = pd.DataFrame(data, columns=['Password'])
    return df

def read_single_file(folder, file_number):
    # Get the list of files in the folder
    files = [f for f in os.listdir(folder) if f.endswith('.txt')]

    # Check if the file number is valid
    if file_number < 1 or file_number > len(files):
        print("Invalid file number. Please choose a number between 1 and", len(files))
        return None

    # Read the selected file
    selected_file = files[file_number - 1]
    file_path = os.path.join(folder, selected_file)

    data = []

    if verbose > 0: print('Opening file: ', file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            email, password = split_email_and_password(line)
            if email is not None and password is not None:
                data.append([password])
                if verbose > 1: print(f"email: {email} password: {password}")
        
        # Create a pandas DataFrame
        df = pd.DataFrame(data, columns=['Password'])
        return df
    
def split_email_and_password(string):
    # Regular expression to capture email and password
    pattern = r"([^:]+)@([^:]+):(.+)"
    
    # Search for matches
    match = re.match(pattern, string)
    
    if match:
        email = match.group(1) + "@" + match.group(2)
        password = match.group(3)
        return email, password
    else:
        return None, None

def extract():
    folder = input("Enter the folder name")

    df = read_files_in_folder(folder + '/data')

    if df is not None:
        print("\nData read:")
        print(df)
        
        df.to_csv(folder + '/extracted_data.csv', index=False)

def mode_select(password, line, mode):
    # Call the mode function
    if mode == 'split_user_email_pass': return getattr(modes, mode)(line)
    return getattr(modes, mode)(password)

if __name__ == '__main__':
    extract()

