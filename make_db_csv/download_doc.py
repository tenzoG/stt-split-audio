import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
import gdown
from docx import Document
from tqdm.auto import tqdm
from fast_antx.core import transfer

sys.path.append('../util')
from google_utils import read_spreadsheet
from db_utils import get_max_db_id
from common_utils import parse_args_and_load_config



def docx_to_txt(docx_path):
    """Convert DOCX to plain text."""
    doc = Document(docx_path)
    return '\n'.join([p.text for p in doc.paragraphs])


def clean_text(text):
    """Remove chapter numbers and clean text to single line."""
    text = re.sub(r'^\d+\s+[ཀ-ྼ་]+པ།', '', text)
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text


def download_etext(gd_url, file_name, etext_dir, docx_dir):
    """Download Google Doc as DOCX and convert to TXT as single line."""
    os.makedirs(etext_dir, exist_ok=True)
    os.makedirs(docx_dir, exist_ok=True)
    
    txt_path = os.path.join(etext_dir, f'{file_name}.txt')
    if os.path.exists(txt_path):
        return txt_path
    
    docx_url, _ = os.path.split(gd_url)
    docx_url = os.path.join(docx_url, 'export?format=docx')
    
    docx_path = os.path.join(docx_dir, f'{file_name}.docx')
    try:
        gdown.download(docx_url, output=docx_path, quiet=False, fuzzy=True)
        text = docx_to_txt(docx_path)
        text = clean_text(text)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Downloaded: {file_name}")
        return txt_path
    except Exception as e:
        print(f"✗ Failed to download {file_name}: {e}")
        return None


def extract_tsv_text(dataframe, column_name):
    """Extract text from dataframe column and format for annotation transfer."""
    predicted_text = dataframe[column_name].tolist()
    
    for i, text in enumerate(predicted_text):
        predicted_text[i] = predicted_text[i].replace(" ", "_")
    
    formatted_text = "\n".join(" ".join(predicted_text).split())
    print("Extracted text from CSV...")
    return formatted_text


def get_original_text(text_file_path):
    """Read original text file and clean unwanted characters."""
    target_text = Path(text_file_path).read_text(encoding="utf-8")
    target_text = target_text.replace(""", "").replace(""", "")
    print("Extracted text from etext...")
    return target_text


def transfer_text(original_text_path, predicted_csv_path, file_name, 
                 column_name='inference_transcript'):
    """Transfer annotation using fast_antx - SIMPLE, no modifications."""
    csv_data = pd.read_csv(predicted_csv_path, sep=",")
    
    csv_data = csv_data[csv_data['file_name'].str.contains(file_name, na=False)].copy()
    csv_data.sort_values(by=['file_name'], inplace=True)
    csv_data.reset_index(drop=True, inplace=True)
    
    if csv_data.empty:
        print(f"⚠ No data found for {file_name}")
        return None, 'No data'
    
    source_text = extract_tsv_text(csv_data, column_name)
    target_text = get_original_text(original_text_path)
    
    annotation = [["segment", "(\n)"]]
    transferred_text = transfer(source_text, annotation, target_text).split("\n")
    
    original_length = len(csv_data)
    transferred_length = len(transferred_text)
    
    if transferred_length > original_length:
        transferred_text = transferred_text[:original_length]
        status = f'Truncated {transferred_length - original_length}'
    elif transferred_length < original_length:
        transferred_text = transferred_text + [''] * (original_length - transferred_length)
        status = f'Padded {original_length - transferred_length}'
    else:
        status = 'Normal'
    
    # Just use what fast_antx gave us
    csv_data[column_name] = transferred_text
    
    # Add text length for analysis
    csv_data['text_length'] = [len(t) for t in transferred_text]
    
    return csv_data, status


def main(config):
    """Main execution function."""
    print("="*60)
    print("STT Text Transfer - IDENTIFY PROBLEM FILES")
    print("="*60)
    
    dept = config['DEPARTMENT']
    from_id = config['FROM_ID']
    to_id = config['TO_ID']
    id_col = config['ID_COL']
    sr_col = config['SR_COL']
    sheet_id = config['SHEET_ID']
    group_id = config['GROUP_ID']
    
    audio_text_link_col = config.get('AUDIO_TEXT_LINK_COL', 'Audio text link')
    etext_dir = config.get('ETEXTS_DIR', '../data/etexts')
    docx_dir = config.get('DOCX_DIR', '../data/docx')
    predicted_csv = config.get('PREDICTED_CSV', f'../data/{dept}_{from_id}_to_{to_id}.csv')
    column_name = config.get('COLUMN_NAME', 'inference_transcript')
    duration_col = config.get('DURATION_COL', 'audio_duration')
    
    print(f"\n[Step 1/5] Reading Google Sheet...")
    sheet_name = config.get('SHEET_NAME', None)
    df_sheet = read_spreadsheet(sheet_id=sheet_id, sheet_name=sheet_name)
    print(f"✓ Loaded {len(df_sheet)} rows")
    
    print(f"\n[Step 2/5] Saving spreadsheet data as TSV...")
    tsv_output = f"stt_{dept.lower()}_from_yt.tsv"
    df_sheet.to_csv(tsv_output, index=False, sep="\t")
    print(f"✓ Saved to {tsv_output}")
    
    print(f"\n[Step 3/5] Downloading etexts for SrNo {from_id} to {to_id}...")
    file_ids_to_process = []
    
    for _, row in df_sheet.iterrows():
        sr_no = row[sr_col]
        file_id = row[id_col]
        
        if from_id <= sr_no <= to_id:
            gd_url = row.get(audio_text_link_col, None)
            
            if gd_url and isinstance(gd_url, str):
                print(f"Processing: {file_id} (Sr.No {sr_no})")
                txt_path = download_etext(gd_url, file_id, etext_dir, docx_dir)
                if txt_path:
                    file_ids_to_process.append(file_id)
    
    print(f"\n✓ Downloaded {len(file_ids_to_process)} etexts")
    
    if not file_ids_to_process:
        print("✗ No files to process!")
        return
    
    print(f"\n[Step 4/5] Transferring text for {len(file_ids_to_process)} files...")
    

    if not os.path.exists(predicted_csv):
        print(f"✗ Predicted CSV not found: {predicted_csv}")
        return
    
    print(f"✓ Found predicted CSV: {predicted_csv}")
    
    transferred_dfs = []
    
    for file_id in tqdm(file_ids_to_process, desc="Transferring"):
        etext_path = os.path.join(etext_dir, f'{file_id}.txt')
        
        if os.path.exists(etext_path):
            transfer_df, status = transfer_text(
                etext_path,
                predicted_csv,
                file_id,
                column_name
            )
            
            if transfer_df is not None:
                print(f"  {file_id}: {status} ({len(transfer_df)} rows)")
                transferred_dfs.append(transfer_df)
        else:
            print(f"  ✗ Skipping {file_id} - etext not found")
    
    if not transferred_dfs:
        print("\n✗ No text transfers completed!")
        return
    
    df_final = pd.concat(transferred_dfs, ignore_index=True)
    print(f"\n✓ Combined {len(transferred_dfs)} files → {len(df_final)} total rows")
    
    print(f"\n[Step 5/5] Preparing final upload CSV...")
    
    df_final['state'] = 'transcribing'
    df_final['group_id'] = group_id
    df_final = df_final.sort_values('file_name').reset_index(drop=True)
    
    last_db_id = get_max_db_id()
    df_final['id'] = range(last_db_id + 1, last_db_id + 1 + len(df_final))
    df_final.fillna('', inplace=True)
    
    # Save full data with text_length for analysis
    analysis_file = f'../data/stt_{dept.lower()}_analysis.csv'
    df_final.to_csv(analysis_file, index=False)
    
    # Create final upload CSV (without text_length)
    final_columns = ['file_name', 'url', column_name, duration_col, 'state', 'group_id', 'id']
    df_upload = df_final[final_columns].copy()
    
    output_file = f'../data/stt_{dept.lower()}_upload_new.csv'
    df_upload.to_csv(output_file, index=False)
    
    # ANALYZE PROBLEM FILES
    print("\n" + "="*60)
    print("PROBLEM FILES ANALYSIS")
    print("="*60)
    
    problem_files = df_final[df_final['text_length'] > 500]
    
    if len(problem_files) > 0:
        print(f"\n⚠ Found {len(problem_files)} segments with >500 characters:")
        print("\nFiles that need manual review:")
        
        problem_summary = problem_files.groupby('file_name').agg({
            'text_length': ['count', 'max', 'mean']
        }).round(0)
        
        print(problem_summary)
        
        # Save problem files list
        problem_list_file = f'../data/stt_{dept.lower()}_PROBLEM_FILES.txt'
        with open(problem_list_file, 'w') as f:
            f.write("SEGMENTS WITH >500 CHARACTERS (NEED MANUAL REVIEW)\n")
            f.write("="*60 + "\n\n")
            for _, row in problem_files.iterrows():
                f.write(f"{row['file_name']}: {row['text_length']} chars (duration: {row[duration_col]}s)\n")
        
        print(f"\n✓ Problem files list saved to: {problem_list_file}")
    else:
        print("\n✓ No segments with >500 characters found!")
    
    print("\n" + "="*60)
    print("Pipeline Completed!")
    print("="*60)
    print(f"Output: {output_file}")
    print(f"Analysis: {analysis_file}")
    print(f"Records: {len(df_final)}")
    print("="*60)


if __name__ == "__main__":
    config = parse_args_and_load_config()
    main(config)