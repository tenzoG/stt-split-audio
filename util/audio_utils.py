import os
import subprocess
from pathlib import Path

import librosa
import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm import tqdm

upper_limit = 8
lower_limit = 2

# [Reason] Keep HF credentials out of source; util/.env already holds other secrets
load_dotenv(Path(__file__).resolve().parent / ".env")


def convert_to_wav_inplace(input_file):
    """
    Convert audio file to PCM-encoded WAV format in place using ffmpeg.
    Args:
    input_file (str): path to full audio file
    """
    try:
        # Generate temporary file path
        temp_file = input_file.replace(".wav", "_temp.wav")
        
        # Run ffmpeg command
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_file, "-ac", "1", "-ar", "16000", temp_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Replace the original file with the temporary file
        os.replace(temp_file, input_file)
        print(f"Converted {input_file} to PCM-encoded WAV format.")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e.stderr.decode()}")
        raise

def sec_to_millis(sec):
    return sec * 1000


def frame_to_sec(frame, sr):
    return frame / sr


def sec_to_frame(sec, sr):
    return sec * sr


HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.5,
    "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 2.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}

# [Reason] pyannote.audio 3.3.2 uses use_auth_token; 4.x renamed it to token
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if not _hf_token:
    raise ValueError(
        "Missing Hugging Face token. Set HF_TOKEN in util/.env "
        "or run `huggingface-cli login`."
    )

pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=_hf_token,
)
pipeline.instantiate(HYPER_PARAMETERS)


def save_segment(segment, folder, prefix, id, start_ms, end_ms, dept):
    os.makedirs(f"../data/{dept}_after_split/{folder}", exist_ok=True)
    segment.export(
        f"../data/{dept}_after_split/{folder}/{prefix}_{id:04}_{int(start_ms)}_to_{int(end_ms)}.wav",
        format="wav",
        parameters=["-ac", "1", "-ar", "16000"],
    )


def delete_file(file):
    os.remove(file)


def split_audio(audio_file, output_folder, dept):
    """splits the full audio file into segments based on
    Voice Activity Detection
    librosa split based on volume and
    blind chop to fit the range of upper_limit to lower_limit

    Args:
        audio_file (str): path to full audio file
        output_folder (str): where to store the split segments
        dept (str): department code used in the output path
    """
    # [Reason] Always create output dir before writing so collect_segments never hits FileNotFoundError
    output_dir = f"../data/{dept}_after_split/{output_folder}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[split_audio] input audio path: {audio_file}")
    print(f"[split_audio] output directory: {output_dir}")

    convert_to_wav_inplace(audio_file)
    vad = pipeline(audio_file)
    vad_timeline = list(vad.get_timeline().support())
    print(f"[split_audio] pyannote VAD spans: {len(vad_timeline)}")

    # [Reason] Surface empty VAD results instead of silently producing no files
    if len(vad_timeline) == 0:
        print(
            f"[split_audio] WARNING: pyannote produced zero segments for {audio_file}. "
            f"Created empty output directory: {output_dir}"
        )
        return

    original_audio_segment = AudioSegment.from_file(audio_file)
    original_audio_ndarray, sampling_rate = torchaudio.load(audio_file)
    original_audio_ndarray = original_audio_ndarray[0]
    counter = 1
    created_files = []

    for vad_span in vad_timeline:
        vad_segment = original_audio_segment[
            sec_to_millis(vad_span.start) : sec_to_millis(vad_span.end)
        ]
        vad_span_length = vad_span.end - vad_span.start
        if vad_span_length >= lower_limit and vad_span_length <= upper_limit:
            save_segment(
                segment=vad_segment,
                folder=output_folder,
                prefix=output_folder,
                id=counter,
                start_ms=sec_to_millis(vad_span.start),
                end_ms=sec_to_millis(vad_span.end),
                dept=dept
            )
            created_path = (
                f"{output_dir}/{output_folder}_{counter:04}_"
                f"{int(sec_to_millis(vad_span.start))}_to_{int(sec_to_millis(vad_span.end))}.wav"
            )
            created_files.append(created_path)
            print(
                f"{counter} {vad_span_length:.2f} {sec_to_millis(vad_span.start):.2f} {sec_to_millis(vad_span.end):.2f} vad"
            )
            counter += 1
        elif vad_span_length > upper_limit:
            non_mute_segment_splits = librosa.effects.split(
                original_audio_ndarray[
                    int(sec_to_frame(vad_span.start, sampling_rate)) : int(
                        sec_to_frame(vad_span.end, sampling_rate)
                    )
                ],
                top_db=30,
            )
            for split_start, split_end in non_mute_segment_splits:
                segment_split = original_audio_segment[
                    sec_to_millis(
                        vad_span.start + frame_to_sec(split_start, sampling_rate)
                    ) : sec_to_millis(
                        vad_span.start + frame_to_sec(split_end, sampling_rate)
                    )
                ]
                segment_split_duration = (
                    vad_span.start + frame_to_sec(split_end, sampling_rate)
                ) - (vad_span.start + frame_to_sec(split_start, sampling_rate))
                if (
                    segment_split_duration >= lower_limit
                    and segment_split_duration <= upper_limit
                ):
                    start_ms = sec_to_millis(
                        vad_span.start + frame_to_sec(split_start, sampling_rate)
                    )
                    end_ms = sec_to_millis(
                        vad_span.start + frame_to_sec(split_end, sampling_rate)
                    )
                    save_segment(
                        segment=segment_split,
                        folder=output_folder,
                        prefix=output_folder,
                        id=counter,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        dept=dept
                    )
                    created_files.append(
                        f"{output_dir}/{output_folder}_{counter:04}_{int(start_ms)}_to_{int(end_ms)}.wav"
                    )
                    print(
                        f"{counter} {segment_split_duration:.2f} {start_ms:.2f} {end_ms:.2f} split"
                    )
                    counter += 1
                elif segment_split_duration > upper_limit:
                    chop_length = segment_split_duration / 2
                    while chop_length > upper_limit:
                        chop_length = chop_length / 2
                    for j in range(int(segment_split_duration / chop_length)):
                        start_ms = sec_to_millis(
                            vad_span.start
                            + frame_to_sec(split_start, sampling_rate)
                            + chop_length * j
                        )
                        end_ms = sec_to_millis(
                            vad_span.start
                            + frame_to_sec(split_start, sampling_rate)
                            + chop_length * (j + 1)
                        )
                        segment_split_chop = original_audio_segment[
                            start_ms:end_ms
                        ]
                        save_segment(
                            segment=segment_split_chop,
                            folder=output_folder,
                            prefix=output_folder,
                            id=counter,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            dept=dept
                        )
                        created_files.append(
                            f"{output_dir}/{output_folder}_{counter:04}_{int(start_ms)}_to_{int(end_ms)}.wav"
                        )
                        print(
                            f"{counter} {chop_length:.2f} {start_ms:.2f} {end_ms:.2f} chop"
                        )
                        counter += 1

    segments_generated = len(created_files)
    print(f"[split_audio] number of segments generated: {segments_generated}")
    if segments_generated == 0:
        # [Reason] VAD found speech but all spans were filtered out by length limits
        print(
            f"[split_audio] WARNING: no segments written for {audio_file} "
            f"(VAD spans={len(vad_timeline)} but none within "
            f"{lower_limit}-{upper_limit}s after split/chop). "
            f"Output directory still created: {output_dir}"
        )
    else:
        print(f"[split_audio] created files:")
        for path in created_files:
            print(f"  - {path}")


def split_audio_files(prefix, ext, audio_dir, dept):
    """
    Process and split all audio files with a specific prefix and extension.

    Args:
        prefix (str): Prefix for the filenames.
        ext (str): File extension of the audio files.
    """
    stt_files = [
        filename
        for filename in os.listdir(audio_dir)
        if filename.startswith(prefix)
        and os.path.isfile(os.path.join(audio_dir, filename))
    ]
    # print(stt_files)
    for stt_file in tqdm(stt_files):
        # print(stt_file)
        stt_file = stt_file.split(".")[0]
        split_audio(audio_file=f"{audio_dir}/{stt_file}.{ext}", output_folder=stt_file, dept=dept)
        # delete_file(file=f"./{stt_folder}/{stt_folder}.wav")


def convert_mp3_to_wav(mp3_file, output_folder):
    # Ensure output folder exists, create if necessary
    os.makedirs(output_folder, exist_ok=True)
    
    # Load MP3 file
    audio = AudioSegment.from_mp3(mp3_file)
    
    # Define output file path (replace .mp3 with .wav)
    wav_file = os.path.join(output_folder, os.path.splitext(os.path.basename(mp3_file))[0] + ".wav")
    
    # Export audio to WAV format
    audio.export(wav_file, format="wav")
    
    print(f"Converted {mp3_file} to {wav_file}")

def batch_convert_mp3_to_wav(input_dir, output_dir):
    # Ensure output directory exists, create if necessary
    os.makedirs(output_dir, exist_ok=True)
    
    # List all files in the input directory
    files = os.listdir(input_dir)
    
    # Filter only .mp3 files
    mp3_files = [f for f in files if f.endswith(".mp3")]
    
    print(mp3_files)
    # Iterate over each MP3 file and convert to WAV
    for mp3_file in mp3_files:
        mp3_path = os.path.join(input_dir, mp3_file)
        convert_mp3_to_wav(mp3_path, output_dir)


def convert_to_16K(input_file, output_file):
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping conversion.")
        return False
    
    # Construct the ffmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_file,
        "-f", "wav",            # Force input format as WAV
        "-bitexact",            # Preserve exact precision
        "-acodec", "pcm_s16le", # Audio codec: PCM signed 16-bit little-endian
        "-ac", "1",             # Mono channel
        "-ar", "16000",         # 16 kHz sample rate
        output_file,
        "-y"                    # Overwrite output file if it exists
    ]
    
    # Join the ffmpeg command into a single string for subprocess
    ffmpeg_command_str = " ".join(ffmpeg_command)
    
    try:
        # Run ffmpeg command using subprocess
        subprocess.run(ffmpeg_command_str, shell=True, check=True, capture_output=True)
        print(f"Conversion successful: {input_file} -> {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return False

def convert_all_to_16K(input_directory, output_directory_16K):
    if not os.path.exists(output_directory_16K):
        os.makedirs(output_directory_16K)

    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory_16K, filename)
            convert_to_16K(input_file, output_file)


def extract_audio(video_file, audio_file):
    command = [
        "ffmpeg",
        "-i", video_file,
        "-q:a", "0",
        "-map", "a",
        audio_file
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Extracted audio to {audio_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to extract audio from {video_file}. Error: {e}")
