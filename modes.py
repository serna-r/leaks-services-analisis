import re

# Extract hex mode
def hex_to_ascii(match):
    # Extraer la cadena hexadecimal dentro de $HEX[]
    hex_str = match.group(1)
    # Convertir la cadena hexadecimal a bytes
    bytes_obj = bytes.fromhex(hex_str)
    # Convertir los bytes a una cadena ASCII
    ascii_str = bytes_obj.decode('utf-8', errors='ignore')
    ascii_str = ascii_str.replace("\x00", "")
    return ascii_str

def extract_hex(text):
    # Expresión regular para detectar el formato $HEX[3132333435360000]
    hex_pattern = re.compile(r'\$HEX\[([0-9A-Fa-f]+)\]')
    # Reemplazar todas las coincidencias en el texto
    return hex_pattern.sub(hex_to_ascii, text)

def split_user_email_pass(line):
    # Eliminate user that is before the first ":"
    emailpass = line.split(':', 1)[1]

    # Regular expression to capture email and password
    pattern = r"([^:]+)@([^:]+):(.+)"
    
    # Search for matches
    match = re.match(pattern, emailpass)
    
    if match:
        password = match.group(3)
        if '$HEX' in password: password = extract_hex(password)
        return password
    else:
        return None