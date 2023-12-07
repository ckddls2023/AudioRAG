import json
import subprocess

query_json_files = [
  #'./data/json_files/AudioSet/val.json',
  './data/json_files/Clotho/val.json',
  './data/json_files/Clotho/train.json',
  './data/json_files/AudioSet/train.json',
]
base_command = "~/dev/DL/llama.cpp/main -t 10 -ngl 40 -m ~/.cache/checkpoints/openorca-platypus2-13b.Q5_K_S.gguf \
-c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p \"### Instruction: Your role is break down complex 'sentence' to easy sentences with delimeter ','. \
When you break down sentences, write sentences as short one. Drop sentence meaning less such as 'needs oil','pops','gravel path','cutting'. \
This is examples of break down sentences. \
'Birds chirp and a pop occurs before a man speaks' can be broken down into two sentences. 'Bird chirp, A man speaks'. \
'A vehicle revving in the distance followed by a man shouting in the distance then a vehicle engine running idle before accelerating and driving off' can be broken down into two sentences. 'A vehicle revving, man shouting, driving off'. \
'A man speaks and sweeps a surface' can be break down into two sentences. 'A man speaks, sweeps a surface'. \
'A motor runs and fades as an adult man speaks' can be break down two sentences. 'A engine runs, a man speaks'. \
'Repeating clicking gets faster until making a continues high pitch sound which slows and repeats' cannot break down. 'Clicks accelerate to a high-pitch and repeat'. \
'An engine increases in speed as a horn honks and a man speaks' can be break down three sentences. 'accelerate, A horn sounds, A man speaks'.\
'In the background, a group of people indistinctly chatter.' cannot break down. 'A group of people chatter'. \
'A dog whimpers and bark' cannot break down. 'A dog barks'. \
'fingers scrape on a metal tin, and have no apparent pattern' cannot break down. 'scrape metal noises'. \
'An audience gives applause and a man laughs before speaking' can be break down three sentences. 'An audience applause, man laughs, man speaks'. \
'Blades spinning' cannot break down. 'Blades spinning'. \
Please break down '{}'. Answer results briefly in one sentence with comma. ### Response: '\""

timeout_duration = 30  # For example, 30 seconds
max_retries = 3  # Maximum number of retries

for json_file in query_json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    num_captions = data["num_captions_per_audio"]
    for entry in data["data"]:
        if num_captions > 1:
            sorted_captions = sorted(entry["caption"], key=len)
            median_index = len(sorted_captions) // 2
            caption = sorted_captions[median_index] # Choose median length
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
            if len(parts) == 2:
                tags = parts[1].strip().replace('"', '').replace("'", "").split(',') # Remove any leading/trailing whitespace
                entry["tag"] = tags
                print(tags)
            else:
                print(f"{entry['audio'] =} are not correctly processed")
                print(result.stdout)
                
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
