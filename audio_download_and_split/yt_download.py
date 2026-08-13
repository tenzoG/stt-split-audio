import sys
import os
import shutil
import subprocess

sys.path.append('../util')

from google_utils import read_spreadsheet
from files_utils import collect_segments
from audio_utils import split_audio_files
from common_utils import parse_args_and_load_config
from yt_dlp_utils import (
    PipelineSummary,
    directory_has_verified_audio,
    download_audio,
    validate_downloaded_file,
)


def main(config):
    """Download YouTube audio for a sheet range, then split/collect/upload verified files."""
    # Read configuration from the loaded JSON
    dept = config['DEPARTMENT']
    from_id = config['FROM_ID']
    to_id = config['TO_ID']
    id_col = config['ID_COL']
    link_col = config['LINK_COL']
    sr_col = config['SR_COL']
    prefix = config['PREFIX']
    segment_dir = config['SEGMENT_DIR']
    download_audio_dir = config['DOWNLOAD_AUDIO_DIR']
    file_format = config['FILE_FORMAT']
    s3_bucket = config['S3_BUCKET']
    sheet_id = config['SHEET_ID']

    # Read the spreadsheet
    df = read_spreadsheet(sheet_id=sheet_id)

    # --- Step 1: Download audio from YouTube links ---
    summary = PipelineSummary()
    successful_paths = []

    for index, row in df.iterrows():
        file_id = row[id_col]
        yt_url = row[link_col]
        sr_no = row[sr_col]

        if not (sr_no >= from_id and sr_no <= to_id):
            continue

        url_value = "" if yt_url is None else str(yt_url).strip()
        if not url_value:
            print(f"\n>>> Skipping {file_id}: empty URL")
            summary.record_skip()
            continue

        print(f"\n>>> Downloading: {file_id} — {url_value}")
        result = download_audio(
            url=url_value,
            output_dir=download_audio_dir,
            file_stem=str(file_id),
            config=config,
            audio_format=file_format,
        )
        summary.record(result)

        if not result.success:
            print(f"[FAILED] Download for {file_id} ({url_value}): {result.reason}")
            continue

        # [Reason] Gate splitting on a second validation pass for the exact file
        ok, reason = validate_downloaded_file(result.filepath, expected_ext=file_format)
        if not ok:
            print(f"[FAILED] Post-download validation for {file_id}: {reason}")
            summary.downloaded = max(0, summary.downloaded - 1)
            summary.failed += 1
            continue

        successful_paths.append(result.filepath)

    summary.print_report()

    # [Reason] Never call split_audio_files unless at least one verified audio exists
    ready, ready_reason = directory_has_verified_audio(
        download_audio_dir,
        prefix=prefix,
        expected_ext=file_format,
    )
    if not successful_paths or not ready:
        print("✗ Skipping split_audio_files()")
        print("Reason:")
        print(ready_reason if not ready else "No successful downloads in this run.")
        # [Reason] Non-zero exit only when every in-range video failed
        if summary.failed > 0 and summary.downloaded == 0:
            sys.exit(1)
        return

    # --- Step 2: Split the audio files ---
    print(f"\n✓ Verified {len(successful_paths)} download(s); starting split_audio_files()")
    split_audio_files(prefix, file_format, download_audio_dir, dept)

    after_split_dir = f'../data/{dept}_after_split'
    if not os.path.isdir(after_split_dir) or not os.listdir(after_split_dir):
        raise RuntimeError(
            f"No split output found in '{after_split_dir}'. "
            f"Check that PREFIX ('{prefix}') matches the downloaded filenames in '{download_audio_dir}'."
        )

    # --- Step 3: Collect the audio segments ---
    # [Reason] Clear stale segments so S3 upload only contains this run's output
    if os.path.exists(segment_dir):
        shutil.rmtree(segment_dir)
    collect_segments(prefix, after_split_dir, segment_dir)

    if not os.path.isdir(segment_dir) or not os.listdir(segment_dir):
        raise RuntimeError(
            f"'{segment_dir}' is empty after collect_segments — nothing to upload. "
            f"Check collect_segments() logic against contents of '{after_split_dir}'."
        )

    # --- Step 4: Upload the collected segments to S3 ---
    env = os.environ.copy()
    # [Reason] Work around AWS CLI stream-rewind checksum bug on large recursive uploads
    env["AWS_REQUEST_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"
    # [Reason] Match bucket region to avoid redirect/retry overhead
    env["AWS_DEFAULT_REGION"] = "ap-south-1"

    try:
        subprocess.run(
            f'aws s3 cp {segment_dir} {s3_bucket} --recursive --region ap-south-1',
            shell=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] S3 upload from '{segment_dir}' to '{s3_bucket}': {e}")
        raise

    print(f"Pipeline complete for department '{dept}', IDs {from_id}-{to_id}.")


if __name__ == "__main__":
    # Parse arguments and load config
    config = parse_args_and_load_config()

    # Run the main pipeline logic
    main(config)
