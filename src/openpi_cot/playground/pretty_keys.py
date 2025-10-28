import re
import ast

def group_model_components(input_file_path, output_file_path):
    """
    Reads a string representation of a list from an input file,
    groups related component names by replacing numbered parts with a variable,
    and writes the unique grouped names to an output file with a blank line
    between each name.

    Args:
        input_file_path (str): The path to the input .txt file.
        output_file_path (str): The path where the output .txt file will be saved.
    """
    try:
        with open(input_file_path, 'r') as f:
            # Read the entire file content, which is a single line string of a list.
            file_content = f.read()
            # Use ast.literal_eval to safely parse the string into a Python list.
            component_names = ast.literal_eval(file_content)
    except (FileNotFoundError, ValueError, SyntaxError) as e:
        print(f"Error: Could not read or parse the file '{input_file_path}'. Please ensure it is a valid list. Details: {e}")
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
        # Write each name followed by two newlines to add a blank line.
        for name in sorted(list(grouped_names)):
            f.write(name + '\n\n')

    print(f"Processing complete. Grouped components saved to '{output_file_path}'.")

if __name__ == '__main__':
    # Define the input and output file names.
    input_file = 'pi05_keys.txt'
    output_file = 'grouped_pi05_keys.txt'
    
    # Run the function.
    group_model_components(input_file, output_file)