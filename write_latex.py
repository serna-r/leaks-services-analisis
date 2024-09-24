from retrieve_stats import get_leak_types, get_count, get_password_length_mean, get_count_and_probabilities

def write_latex_table(data, output_path):
    with open(output_path, 'w') as file:
        file.write("\\begin{table*}[h!]\n")
        file.write("\\centering\n")
        file.write("\\begin{tabular}{|l|l|c|c|c|c|}\n")  # Added a new 'l' for the Category column
        file.write("\\hline\n")
        file.write("\\textbf{Source} & \\textbf{Category} & \\textbf{Date} & \\textbf{Number of Users} & \\textbf{Mean Length} & \\textbf{Mean Score} \\\\\n")
        file.write("\\hline\n")
        # Loop through the data dictionary and fill the table with appropriate values
        for source, stats in data.items():
            category, num_users, mean_length, mean_score  = stats  # Unpack the category
            # The Date column is left empty as per your table structure
            file.write(f"{source} & {category} &  & {num_users} & {mean_length:.2f} & {mean_score:.2f} \\\\\n")
        file.write("\\hline\n")
        file.write("\\end{tabular}\n")
        file.write("\\caption{Summary of data breaches with user information.}\n")
        file.write("\\end{table*}\n")

def calculate_mean_score(probability_dist):
    """Calculate the weighted mean score based on the probabilities."""
    scores = [0,1,2,3,4]
    mean_score = sum(score * prob for score, prob in zip(scores, probability_dist))
    return mean_score

def get_latex_table(output_path):
    leaks_file='leak_types.txt'
    # Get leaks
    leak_types = get_leak_types(leaks_file)
    
    # Dictionary to store the results for each leak
    leak_data = {}
    
    for leak, category in leak_types:
        # Get leak file name
        leak_file = './leaks/' + leak + '/Stats.txt'
        
        try:
            # Open file and get stats
            with open(leak_file, 'r') as f:
                # Read file content
                data = f.read()
                
                # Get number of users, mean password length, and score probabilities
                count = get_count(data)
                mean_length = get_password_length_mean(data)
                count_list, probability_dist = get_count_and_probabilities(data)
                
                # Calculate the mean score using the score probabilities
                mean_score = calculate_mean_score(probability_dist)
                
                # Add to the dictionary (number of users, mean length, mean score)
                leak_data[leak] = ( category, count, mean_length, mean_score)
        
        except FileNotFoundError:
            print(f"File not found: {leak_file}")
            continue

    # Write the LaTeX table to the output path
    write_latex_table(leak_data, output_path)

if __name__ == '__main__':
    # Define the output path for the LaTeX table
    output_path = 'latex/distribution_resume_table.tex'
    
    # Generate the LaTeX table
    get_latex_table(output_path)