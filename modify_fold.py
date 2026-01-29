import json
import sys

def prepend_to_paths(json_file, prepend_string="/cluster/projects/vc/data/mic/open/Prostate/", output_file=None):
    """
    Prepend a string to t2w, adc, and hbv paths in a JSON file.
    
    Args:
        json_file: Path to input JSON file
        prepend_string: String to prepend to the paths
        output_file: Path to output JSON file (if None, overwrites input file)
    """
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Process both 'train' and 'validation' sections
    for section in ["test", 'training', 'validation']:
        if section in data:
            for item in data[section]:
                # Prepend to t2w, adc, and hbv paths
                for key in ["image", "prostate_pred", "prostate_prob_map", "prostate", "zones", "pca"]:
                    if key in item:
                        if type(item[key]) == list:
                            new_list = []
                            for list_item in item[key]:
                                 new_list.append(prepend_string + list_item)
                            item[key] = new_list
                        else:
                            item[key] = prepend_string + item[key]
        else:
            print("No section: ", section, " in data.")
    
    # Write the modified data back to file
    output_path = output_file if output_file else json_file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Successfully updated paths in {output_path}")

if __name__ == "__main__":
    prepend_to_paths(f"json_datalists/picai/all_samples.json", "/cluster/projects/vc/data/mic/open/Prostate/", f"json_datalists/picai/all_samples_a.json")

    #for i in range(5):
        #prepend_to_paths(f"json_datalists/picai/fold_{i}.json", "/cluster/projects/vc/data/mic/open/Prostate/", f"json_datalists/picai/fold_{i}_abs.json")