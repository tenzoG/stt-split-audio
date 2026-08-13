import sys
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
    """Download YouTube audio using cookie auth, then split only verified files."""
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
    # [Reason] COOKIES_PATH is picked up by get_yt_dlp_config() for backward compatibility

    # Read the spreadsheet
    df = read_spreadsheet(sheet_id=sheet_id)

    summary = PipelineSummary()
    successful_paths = []

    for index, row in df.iterrows():
        file_id = row[id_col]
        yt_url = row[link_col]
        sr_no = row[sr_col]

        if not (sr_no >= from_id and sr_no <= to_id):
            continue

        url_value = "" if yt_url is None else str(yt_url).split('?')[0].strip()
        if not url_value:
            print(f"\n>>> Skipping {file_id}: empty URL")
            summary.record_skip()
            continue

        print(f"\n>>> Processing {file_id}: {url_value}")
        result = download_audio(
            url=url_value,
            output_dir=download_audio_dir,
            file_stem=str(file_id),
            config=config,
            audio_format=file_format,
        )
        summary.record(result)

        if not result.success:
            print(f"Marked failed (will not split): {file_id}")
            continue

        ok, reason = validate_downloaded_file(result.filepath, expected_ext=file_format)
        if not ok:
            print(f"✗ Post-download validation failed for {file_id}")
            print(f"Reason:\n{reason}")
            summary.downloaded = max(0, summary.downloaded - 1)
            summary.failed += 1
            continue

        successful_paths.append(result.filepath)

    summary.print_report()

    ready, ready_reason = directory_has_verified_audio(
        download_audio_dir,
        prefix=prefix,
        expected_ext=file_format,
    )
    if not successful_paths or not ready:
        print("✗ Skipping split_audio_files()")
        print("Reason:")
        print(ready_reason if not ready else "No successful downloads in this run.")
        if summary.failed > 0 and summary.downloaded == 0:
            sys.exit(1)
        return

    print(f"\n✓ Verified {len(successful_paths)} download(s); starting split_audio_files()")
    split_audio_files(prefix, file_format, download_audio_dir, dept)

    # Collect the audio segments
    collect_segments(prefix, f'{dept}_after_split', segment_dir)

    # Upload the collected segments to the S3 bucket
    subprocess.run(f'aws s3 cp {segment_dir} {s3_bucket} --recursive', shell=True)


if __name__ == "__main__":
    # Parse arguments and load config
    config = parse_args_and_load_config()

    # Run the main pipeline logic
    main(config)
