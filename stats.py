import math
from zxcvbn import zxcvbn
import numpy as np

# Function to optimize password processing
def apply_all_in_one(password):
    mask, total = password_mask(password)
    simpleentropy = simple_entropy(password)
    shannonentropy = shannon_entropy(password)
    length = len(password)
    load_top_100(file_path='top100.txt')
    common = is_common_password(password)
    score , guesses = password_strength(password)

    return mask, total, length, simpleentropy, shannonentropy, common, score, guesses

# Apply only one stat
def apply_one(password, metric):
    current_module = __import__(__name__)
    return getattr(current_module, metric)(password) 


def load_top_100(file_path):
    with open(file_path, 'r') as file:
        # Read all lines from the file, excluding the first line
        lines = file.readlines()[1:]  # Skip the first line
        
        # Limit the list to the top 100 passwords
        global top_100_passwords
        top_100_passwords = [line.strip() for line in lines[:100]]
    
    return top_100_passwords

def is_common_password(password):
    # Check if the password is in the list of the top 100
    return password in top_100_passwords

# Function to analyze password complexity
def password_mask(password):
    mask = ''
    if any(c.islower() for c in password): mask = mask + 'l'
    if any(c.isupper() for c in password): mask = mask + 'u'
    if any(c.isnumeric() for c in password): mask = mask + 'd'
    if any(not c.isalnum() for c in password): mask = mask + 's'
    total = True

    # For other languages
    if mask == '': mask = 'z'

    return mask, total

# Calculate simple_entropy using the formula E = L × log(R) / log(2)
def simple_entropy(word):
    word_length = len(word)
    possible_symbols = len(set(word))
    
    if possible_symbols > 1:  # To avoid log(1) or log(0)
        simple_entropy = word_length * (math.log(possible_symbols) / math.log(2))
    else:
        simple_entropy = 0  # If there is only one symbol, simple_entropy is 0
    
    return simple_entropy

# H(X) = - Σ [p(x) * log2(p(x))]
def shannon_entropy(password):
    # Count the frequency of each character in the word
    frequencies = {}
    for char in password:
        if char in frequencies:
            frequencies[char] += 1
        else:
            frequencies[char] = 1
    
    # Calculate Shannon entropy
    entropy = 0.0
    word_length = len(password)
    for frequency in frequencies.values():
        probability = frequency / word_length
        entropy -= probability * math.log2(probability)
    
    return entropy

def password_strength(password):
    result = zxcvbn(password)
    return result.get('score'), float(result.get('guesses'))