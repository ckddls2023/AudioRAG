import csv
import argparse
import json
import librosa

def get_audio_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

# Read the CSV and parse into the desired format
parsed_data = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_path", type=str, required=True)
    args = parser.parse_args()

    csv_file_path = args.csv_file_path
    # Open the CSV file and read its contents
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            # Extract file name and captions
            file_name = row[0]
            captions = row[1:]

            # Create the dictionary format
            entry = {
                "id": file_name,  # Exclude the .wav extension for the ID
                "caption": captions
            }
            parsed_data.append(entry)

    # Iterate over each parsed data entry and add the duration
    for entry in parsed_data:
        # Construct the full file path for the audio file (assuming they are in the same directory)
        file_path = "/drl_nas1/ckddls1321/data/WavCaps/waveforms/CLOTHO_v2.1/" + entry["id"]

        # Get the duration of the audio file using the mock function
        duration = get_audio_duration(file_path)

        # Add the duration and audio path to the entry
        entry["duration"] = duration
        entry["audio"] = file_path

    # Modify the format
    formatted_data = {
        "num_captions_per_audio": 5,
        "data": parsed_data
    }

    # Save the formatted data to a JSON file
    with open(args.csv_file_path.replace(".csv",".json"), 'w') as json_file:
        json.dump(formatted_data, json_file, indent=4)