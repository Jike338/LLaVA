import json
json_path = './playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json'
output_json_path = 'db_example_pt.json'

# json_path = './playground/data/llava_v1_5_mix665k.json'
# output_json_path = 'db_example.json'

# Function to load the original JSON and save the first few items to a new file
def extract_json_items(json_path, output_json_path, num_items=2):
    # Load the original JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract the first 'num_items' items
    extracted_data = data[:num_items]

    # Write the extracted data into a new JSON file, preserving the exact format
    with open(output_json_path, 'w') as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Extracted {num_items} items and saved to {output_json_path}")

if __name__ == '__main__':
    # Extract the first couple of items and save them to a new file
    extract_json_items(json_path, output_json_path, num_items=2)
