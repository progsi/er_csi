import os
import pandas as pd
from youtubesearchpython import Video, ResultMode
import argparse
from tqdm import tqdm
import sys

def parse_csv_files(directory):
    unique_yt_ids = set()

    # Iterate through all CSV files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            # Read CSV using pandas with semicolon separator
            df = pd.read_csv(file_path, sep=';')
            # Assuming 'yt_id' is the column name in CSV
            yt_ids = set(df['yt_id'])
            unique_yt_ids.update(yt_ids)

    return list(unique_yt_ids)

def retrieve_video_metadata(yt_id):
    video = Video.get(f'https://www.youtube.com/watch?v={yt_id}', mode=ResultMode.json, get_upload_date=True)
    return video

def main():
    parser = argparse.ArgumentParser(description='Retrieve YouTube video metadata from CSV files and save to Parquet.')
    parser.add_argument('--directory', default='data/csi_datasets', help='Directory containing CSV files (default: data)')
    args = parser.parse_args()

    unique_yt_ids = parse_csv_files(args.directory)
    print(f"Retrieving metadata for {len(unique_yt_ids)} videos.")

    metadata_list = []

    for i, yt_id in tqdm(enumerate(unique_yt_ids, start=1)):
        try:
            video_metadata = retrieve_video_metadata(yt_id)
            metadata_list.append(video_metadata)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Script interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"Exception {e} at yt_id {yt_id}")

        # Save to Parquet every 50 iterations
        if i % 50 == 0:
            df = pd.DataFrame(metadata_list)
            parquet_filename = f'data/yt_metadata.parquet'
            df.to_parquet(parquet_filename, index=False)
            print(f"Saved metadata to {parquet_filename}")

            # Reset the metadata list for the next batch of iterations
            metadata_list = []

    # Save any remaining metadata
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        parquet_filename = f'data/yt_metadata.parquet'
        df.to_parquet(parquet_filename, index=False)
        print(f"Saved remaining metadata to {parquet_filename}")

if __name__ == "__main__":
    main()
