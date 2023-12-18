import json
import os
from openai import OpenAI

query_json_files = [
  './data/json_files/AudioSet/train.json',
  './data/json_files/AudioSet/val.json',
  './data/json_files/Clotho/train.json',
  './data/json_files/Clotho/val.json',
]

# new

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

query_data = []
for json_file in query_json_files:
    with open(json_file, 'r') as file:
        data = json.load(file)
    for entry in data["data"]:
        keys_to_remove = ['author', 'description', 'download_link', 'category', 'file_name', 'href', 'category', 'title']
        for key in keys_to_remove:
            if key in entry:
                del entry[key]
        captions = entry["caption"]
        if data["num_captions_per_audio"] > 1:
            captions = ""
            for caption in entry["caption"]:
                captions += f"- {caption}\n"
        else:
            captions = "- {}".format(captions)
            
        prompt = f"""
            ### Instruction:
            Your task is to simplify complex audio descriptions into short, clear, distinct tags. When simplifying, focus on capturing the essence of each description in as few words as possible. Omit any unnecessary details or ambiguous phrases. 

            Examples:
            1. Original: "Birds chirp and a pop occurs before a man speaks" 
            Simplified: "Bird chirp, Man speaks"

            2. Original: "A vehicle revving in the distance followed by a man shouting in the distance then a vehicle engine running idle before accelerating and driving off"
            Simplified: "Vehicle revving, Man shouting, Driving off"

            3. Original: "An engine increases in speed as a horn honks and a man speaks"
            Simplified: "Engine accelerates, Horn sounds, Man speaks"

            Your goal is to maintain the core information while making each sentence as straightforward and concise as possible. 

            Extract distinct tags from the following audio captions:\n
            {captions}\n
            ### Response:
        """
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4-0314",
        )
        entry["gpt_tag"] = chat_completion.choices[0].message.content
        print(entry["gpt_tag"])
    # Write the modified data back to the same JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    
    