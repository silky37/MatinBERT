# Path to the file
input_file_path = '/data/user5/workspace/MatinBERT/data/NER_MATSCHOLAR/test.txt'
output_file_path = '/data/user5/workspace/MatinBERT/data/NER_MATSCHOLAR/test_no_labels.txt'

# Open the input file and output file
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    # Process each line in the file
    for line in infile:
        # Split the line by spaces (assuming the format is "word label")
        parts = line.strip().split()
        if len(parts) > 0:
            # Write the token (first part) without the label to the output file
            outfile.write(parts[0] + '\n')
        else:
            # For empty lines, preserve the empty line
            outfile.write('\n')

print(f"Processed file saved to: {output_file_path}")