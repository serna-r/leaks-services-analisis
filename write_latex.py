from retrieve_stats import get_leak_types, get_count, get_password_length_mean, get_count_and_probabilities
import statistics

def write_latex_table(data, output_path):
    with open(output_path, 'w') as file:
        file.write("\\begin{table*}[h!]\n")
        file.write("\\centering\n")
        file.write("\\begin{tabular}{|l|l|c|r|c|c|}\n")  # Added a new 'l' for the Category column
        file.write("\\hline\n")
        file.write("\\textbf{Category} & \\textbf{Source} & \\textbf{Date} & \\textbf{\# Passwords} & \\textbf{Mean Length} & \\textbf{Mean Score} \\\\\n")

        # Get first category
        last_category = ''
        # Get all num users, lengths and scores
        num_users_list, mean_length_list, mean_score_list = [], [], []
        # Loop through the data dictionary and fill the table with appropriate values
        for source, stats in data.items():
            category, date, num_users, mean_length, mean_score  = stats  # Unpack the category
            # Append to correct list
            num_users_list.append(num_users)
            mean_length_list.append(mean_length)
            mean_score_list.append(mean_score)
            # If the category is different print line
            category = category.capitalize()
            if category != last_category:
                file.write("\\hline\n")
                last_category = category
                file.write(f"{category}")
            if category == 'Digitaltool':
                category = 'Digital tool'
            # The Date column is left empty as per your table structure
            file.write(f"&{source.capitalize()} & {date}  & {num_users:,} & {mean_length:.2f} & {mean_score:.2f} \\\\\n")
        file.write("\\hline\n")
        file.write("\\cline{3-6}\n")
        file.write("\\multicolumn{2}{l|}{\\multirow{2}{*}{}}")
        file.write("& \\textbf{Mean}")
        file.write(f"& {statistics.mean(num_users_list):,.0f} & {statistics.mean(mean_length_list):,.2f} & {statistics.mean(mean_score_list):.2f} \\\\\n")
        file.write("\\cline{3-6}\n")
        file.write("\\multicolumn{2}{l|}{}")
        file.write("& \\textbf{Standard deviation}")
        file.write(f"& {statistics.stdev(num_users_list):,.0f} & {statistics.stdev(mean_length_list):,.2f} & {statistics.stdev(mean_score_list):.2f} \\\\\n")
        file.write("\\cline{3-6}")
        file.write("\\end{tabular}\n")
        file.write("\\label{table:dataleaks}")
        file.write("\\caption{Summary of data breaches with user information. \\tiny{* means that it is provided by \cite{dionysiou2021honeygen}}}\n")
        file.write("\\end{table*}\n")

def calculate_mean_score(probability_dist):
    """Calculate the weighted mean score based on the probabilities."""
    scores = [0,1,2,3,4]
    mean_score = sum(score * prob for score, prob in zip(scores, probability_dist))
    return mean_score

def get_latex_table(leaks_file, output_path):
    # Get leaks
    leak_types, dates = get_leak_types(leaks_file, dates=True)
    
    # Dictionary to store the results for each leak
    leak_data = {}
    
    for i, (leak, category) in enumerate(leak_types):
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
                count_list, probability_dist = get_count_and_probabilities(leak)
                
                # Calculate the mean score using the score probabilities
                mean_score = calculate_mean_score(probability_dist)
                
                # Add to the dictionary (number of users, mean length, mean score)
                leak_data[leak] = (category, dates[i], count, mean_length, mean_score)
        
        except FileNotFoundError:
            print(f"File not found: {leak_file}")
            continue
    
    # Order by category
    leak_data_ordered = dict(sorted(leak_data.items(), key=lambda d: d[1]))
    # Write the LaTeX table to the output path
    write_latex_table(leak_data_ordered, output_path)

if __name__ == '__main__':
    # Define the output path for the LaTeX table
    output_path = 'latex/distribution_resume_table.tex'
    
    leaks_file='leak_types.txt'
    
    # Generate the LaTeX table
    get_latex_table(leaks_file ,output_path)