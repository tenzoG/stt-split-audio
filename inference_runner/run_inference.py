import os
import sys

sys.path.append('../util')

from common_utils import get_time_span
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
from common_utils import parse_args_and_load_config

def main(config):
    dept = config['DEPARTMENT']
    segment_dir = config['SEGMENT_DIR']
    target_path = Path(segment_dir)
    rows = []

    for file in tqdm(target_path.glob('*.wav'), total=len(list(target_path.glob('*.wav')))):
        rows.append([
            file.stem, 
            f"https://monlam-ai-stt.s3.amazonaws.com/{file.name}", 
            "",
            get_time_span(str(file.name))
        ])

    df = pd.DataFrame(rows, columns=['file_name', 'url', 'inference_transcript', 'audio_duration'])
    df = df.sort_values('file_name').reset_index(drop=True)
    df.to_csv(f"../data/{dept}.csv", index=False)
    
    print(f"\n✅ Results saved to: ../data/{dept}.csv")
    print(f"Total files processed: {len(df)}")

if __name__ == "__main__":
    config = parse_args_and_load_config()
    main(config)