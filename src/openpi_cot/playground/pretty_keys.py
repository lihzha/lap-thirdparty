import re

def group_model_components(input_file_path, output_file_path):
    """
    Reads model component names from an input file, groups related names
    by replacing numbered parts with a variable, and writes the unique
    grouped names to an output file.

    For example, 'Component_0.bias' and 'Component_1.bias' will both
    become 'Component_X.bias'.

    Args:
        input_file_path (str): The path to the input .txt file.
        output_file_path (str): The path where the output .txt file will be saved.
    """
    try:
        with open(input_file_path, 'r') as f:
            # Assumes one component name per line.
            component_names = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        return

    grouped_names = set()

    for name in component_names:
        # Split the component name by '.' to check each part.
        parts = name.split('.')
        new_parts = []
        for part in parts:
            # Use a regular expression to find parts ending in '_' followed by digits
            # and replace them with '_X'.
            new_part = re.sub(r'_(\d+)$', '_X', part)
            new_parts.append(new_part)
        
        # Rejoin the parts to form the generalized name.
        generalized_name = '.'.join(new_parts)
        grouped_names.add(generalized_name)

    # Sort the unique names alphabetically and save them to the output file.
    with open(output_file_path, 'w') as f:
        for name in sorted(list(grouped_names)):
            f.write(name + '\n')

    print(f"Processing complete. Grouped components saved to '{output_file_path}'.")

if __name__ == '__main__':
    # Define the input and output file names.
    input_file = 'pi05_keys.txt'
    output_file = 'grouped_pi05_keys.txt'
    
    # Run the function.
    group_model_components(input_file, output_file)