import math

# with open('shein\data\Shein.com 30kk.txt','r', encoding='utf-8') as f:
#             for line in f:
#                 print(line)

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

print(shannon_entropy('123456'))
print(shannon_entropy('holahola'))
print(shannon_entropy('$%adf&*jkkl%adskl;'))
print(shannon_entropy('r/learnpython'))