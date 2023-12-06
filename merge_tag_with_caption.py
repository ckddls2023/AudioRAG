import json
import subprocess

query_json_files = [
  #'./data/json_files/AudioSet/val.json',
  #'./data/json_files/Clotho/val.json',
  #'./data/json_files/Clotho/train.json',
  './data/json_files/AudioSet/train.json',
]
base_command = "~/dev/DL/llama.cpp/main -t 10 -ngl 32 -m ~/.cache/checkpoints/openorca-platypus2-13b.Q5_K_S.gguf \
-c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p \"### Instruction: Please break down complex 'sentence' to easy sentences with delimeter ','. \
For example 'Birds chirp and a pop occurs before a man speaks' can be broken down into two sentences. 'Bird chirp, A man speaks'. \
For another example 'A vehicle revving in the distance followed by a man shouting in the distance then a vehicle engine running idle before accelerating and driving off' can be broken down into two sentences. 'A vehicle revving, man shouting, driving off'. \
For other example 'A man speaks and sweeps a surface' can be break down into two sentences. 'A man speaks, sweeps a surface'. \
For simple example 'Blades spinning' cannot be break down. Just write 'Blades spinning'. \
Please break down '{}'. Answer briefly in one sentence with comma. ### Response: \""

timeout_duration = 30  # For example, 30 seconds
max_retries = 3  # Maximum number of retries

for json_file in query_json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    num_captions = data["num_captions_per_audio"]
    for entry in data["data"]:
        if num_captions > 1:
            caption = min(entry["caption"], key=len)  # Choose the shortest caption
        else:
            caption = entry["caption"]  # Single caption
        command = base_command.format(caption)
        attempt = 0
        while attempt < max_retries:
            try:
                # Attempt to run the subprocess with a timeout
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_duration)
                break  # Break the loop if command is successful
            except subprocess.TimeoutExpired:
                print(f"Command timed out for caption: {caption}. Attempt {attempt + 1}/{max_retries}.")
                attempt += 1  # Increment attempt counter

        if attempt == max_retries:
            print(f"Max retries reached for caption: {caption}. Skipping...")
            continue  # Skip to the next entry 

        if result.returncode == 0:
            # Split the output to extract the part after "### Response:"
            parts = result.stdout.split("### Response:")
            if len(parts) > 1:
                tags = parts[1].strip().split(',')  # Remove any leading/trailing whitespace
                entry["tag"] = tags
            else:
                print(f"{entry['audio'] =} are not correctly processed")
                print(result.stdout)
                
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
