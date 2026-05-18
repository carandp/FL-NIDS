import json

# List of your 3 JSON files
json_files = ["dp_budget_client0.json", "dp_budget_client1.json", "dp_budget_client2.json"]

# Open the output CSV file for writing
with open("client_history.csv", "w") as out_file:
    # Write the header row first
    out_file.write("client id,noise multiplier,round,epsilon\n")

    # Process each JSON file
    for file_name in json_files:
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
            
            # Pull top-level values
            client_id = data.get("client_id")
            noise_mult = data.get("noise_multiplier")
            
            # Loop through the history array and append rows directly to the file
            for entry in data.get("history", []):
                round_num = entry.get("round")
                epsilon = entry.get("epsilon")
                
                # Write line break formatted row to file
                out_file.write(f"{client_id},{noise_mult},{round_num},{epsilon}\n")
                
        except FileNotFoundError:
            print(f"Warning: File {file_name} not found. Skipping...")