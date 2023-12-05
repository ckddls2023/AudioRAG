import json
import subprocess

query_json_files = [
  './data/json_files/AudioSet/train.json',
  './data/json_files/AudioSet/val.json',
  './data/json_files/Clotho/train.json',
  './data/json_files/Clotho/val.json',
]
base_command = "~/dev/DL/llama.cpp/main -t 10 -ngl 32 -m ~/.cache/checkpoints/openorca-platypus2-13b.Q5_K_S.gguf \
-c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p \"### Instruction: Please break down complex 'sentence' to easy sentences with delimeter ','. \
For example 'Birds chirp and a pop occurs before a man speaks' can be broken down into two sentences. 'Bird chirp, A man speaks'. \
For another example 'A vehicle revving in the distance followed by a man shouting in the distance then a vehicle engine running idle before accelerating and driving off' can be broken down into two sentences. 'A vehicle revving, man shouting, driving off'. \
For other example 'A man speaks and sweeps a surface' can be break down into two sentences. 'A man speaks, sweeps a surface'. \
For simple example 'Blades spinning' cannot be break down. Just write 'Blades spinning'. \
Please break down '{}'. Answer briefly in one sentence with comma. ### Response: \""

for json_file in query_json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    num_captions = data["num_captions_per_audio"]
    for entry in data["data"]:
        caption = entry["caption"]
        command = base_command.format(caption)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode == 0:
            # Split the output to extract the part after "### Response:"
            parts = result.stdout.split("### Response:")
            if len(parts) > 1:
                response = parts[1].strip()  # Remove any leading/trailing whitespace
                tags = [tag.strip() for tag in response]
                entry["tag"] = tags
                
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
