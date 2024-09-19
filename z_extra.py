import pandas as pd
import re

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

data = []
verbose = 2
file_path = './shein/data/shein.com 30kk.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        email, password = split_email_and_password(line)
        if email is not None and password is not None:
            # Check also for unknown or null passwords
            if (password == 'NULL' or password == 'none' or password == '?' 
                    or password == 'None'):
                continue
            data.append([email, password])
            if verbose > 1: print(f"email: {email} password: {password}")

# Create a pandas DataFrame with all the collected data
df = pd.DataFrame(data, columns=['User', 'Password'])