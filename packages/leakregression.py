from packages.retrieve_stats import get_leak_types

def leakregression(leaks_file='leak_types.txt'):
    """Function to create a regression between password strength in the leaks and their risk value"""
    # Get leaks
    leak_types = get_leak_types(leaks_file)
    leak_names = [x for x,y in leak_types]

    # Get risk for leaks

    print()


    return