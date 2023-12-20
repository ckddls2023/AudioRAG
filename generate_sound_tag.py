import json
import os
import re
from openai import OpenAI
from llama_cpp import Llama

query_json_files = [
  './data/json_files/AudioSet/val.json',
  './data/json_files/Clotho/val.json',
  './data/json_files/Clotho/train.json',
  './data/json_files/AudioSet/train.json',
]

llm = Llama(model_path="/home/ckddls1321/.cache/checkpoints/solar-10.7b-instruct-v1.0.Q5_K_S.gguf", 
            chat_format="llama-2",
            main_gpu=0,
            n_threads=16,            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=49,         # The number of layers to offload to GPU, if you have GPU acceleration available
            n_ctx=4096  # The max sequence length to use - note that longer sequence lengths require much more resources
)  

# client = OpenAI(
#   api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
# )

def remove_parentheses(text):
    # This regex pattern matches anything within parentheses
    pattern = r'\([^)]*\)'
    # Replace the matched text with an empty string
    return re.sub(pattern, '', text)

query_data = []
for json_file in query_json_files:
    print(f"Start to process : {json_file}")
    with open(json_file, 'r') as file:
        data = json.load(file)
    for entry in data["data"]:
        keys_to_remove = ['author', 'description', 'download_link', 'category', 'file_name', 'href', 'category', 'title']
        for key in keys_to_remove:
            if key in entry:
                del entry[key]
        captions = entry["caption"]
        if data["num_captions_per_audio"] > 1:
            captions = entry["caption"][0]
            
        prompt = f"""
            Your task is to extract tags from caption. 
            When extract tag, focus on capturing the essence of each description. 
            Omit any unnecessary or ambiguous phrases. 
            Tag should be more than two words. 
            Extract tag from the following caption. 
            
            This is example of caption and tag.
            Caption: 'Birds chirp and a pop occurs before a man speaks' 
            Tags: 'Bird chirp, Man speaks'
            Caption: '{captions}'
            Tags:
        """
        
        # Chat Completion API
        # Continue making requests until completion_tokens > 0
        if isinstance(entry["tag"], list) or 'extracted tag' in entry['tag']: # Not processed tags only
            completion_tokens = 0
            chat_completion = llm.create_chat_completion(
                messages = [
                    {"role": "user", "content": prompt}
                ]
            )
            # chat_completion = client.chat.completions.create(
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": prompt,
            #         }
            #     ],
            #     model="gpt-4-0314",
            # )
            completion_tokens = chat_completion['usage']['completion_tokens']
            if completion_tokens > 0 and completion_tokens < 40:
                entry["tag"] = remove_parentheses(chat_completion['choices'][0]['message']['content'].replace("[SOLUTION]","").replace("[SOL]","").replace("Tags: ","").replace(": ",""))
            else:
                entry["tag"] = ','.join(entry["tag"]) # convert to comma-seperate tag
            llm.reset()
            print(entry["tag"])
        else:
            entry["tag"] = entry["tag"].strip("'").strip().replace(", ",",").replace(" ,","").replace(", ",",")
    # Write the modified data back to the same JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    
    
