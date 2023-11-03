import json
import subprocess
import argparse
from tqdm import tqdm

# Create the parser
parser = argparse.ArgumentParser(description='Run llama.cpp with provided model and get the summarized sentence.')

# Add the arguments
parser.add_argument('--llama_cpp_path', type=str, required=True, help='The path to the llama.cpp executable')
parser.add_argument('--model_path', type=str, required=True, help='The path to the llama model directory')
parser.add_argument('--json_file', type=str, required=True, help='The path to the llama model directory')

# Parse the arguments
args = parser.parse_args()

# Define the command with the arguments provided
cmd = [
    args.llama_cpp_path,  # Path to the llama.cpp executable
    "-m",
    args.model_path,      # Path to the llama model directory
    "-p",
    "Summarize '''Sentences''' into one. '''Sentences : A dog is barking at a man walking by. A dog barks near by man who is walking''' Merged sentence : ",
    "-n",
    "400",
    "-e"
]

templates = [ # Should use regex..?, Keep it simple
    "Summarized sentence :",
    "summarized sentence :",
    "Summarized sentence:",
    "summarized sentence:",
    "Summarized sentence is :",
    "summarized sentence is :",
    "Summarized sentence is:",
    "summarized sentence is:",
]

with open(args.json_file, "r") as f:
    json_obj = json.load(f)
    for item in tqdm(json_obj["data"]):
        captions = ' '.join(item["caption"])
        cmd[4]=f"Summarize only important or common parts in '''Sentences'''. '''Sentences''' : {captions}'''. Summarized sentence :"
        process = subprocess.run(cmd, capture_output=True, text=True)
        stdout = process.stdout
        # Iterate over templates and find the first one that matches
        for template in templates:
            start_index = stdout.find(template)
            if start_index != -1:
                # Adjust the start index to get the text after the template
                start_index += len(template)
                break
        end_index = stdout.find("[end of text]")
        merged_sentence = stdout[start_index:end_index].strip()
        print(captions)
        print(merged_sentence)
        item["caption"] = merged_sentence

with open(args.json_file.replace("final.json","final_merged.json"), 'w') as json_file:
    json.dump(json_obj, json_file, indent=4)
