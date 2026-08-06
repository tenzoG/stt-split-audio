import re
import json
import argparse


def clean_transcription(text):
    """
    Cleans and normalizes Tibetan transcription text to make it syntactically correct.

    Args:
        text (str): The input transcription text.

    Returns:
        str: The cleaned and normalized transcription text.
    """
    # Replace newline and tab characters with spaces
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.strip()
    
    # Normalize specific Tibetan punctuation and characters
    text = re.sub("༌", "་", text)  # Normalize tsak
    text = re.sub("༎", "།", text)  # Normalize double shae
    text = re.sub("༔", "།", text)
    text = re.sub("༏", "།", text)
    text = re.sub("༐", "།", text)
    text = re.sub("ཽ", "ོ", text)  # Normalize
    text = re.sub("ཻ", "ེ", text)  # Normalize

    # Collapse multiple spaces and Tibetan spaces
    text = re.sub(r"\s+།", "།", text)
    text = re.sub(r"།+", "།", text)
    text = re.sub(r"།", "། ", text)
    text = re.sub(r"\s+་", "་", text)
    text = re.sub(r"་+", "་", text)
    text = re.sub(r"\s+", " ", text)

    # Normalize repetitive sequences of Tibetan characters
    text = re.sub(r"ཧཧཧ+", "ཧཧཧ", text)
    text = re.sub(r'ཧི་ཧི་(ཧི་)+', r'ཧི་ཧི་ཧི་', text)
    text = re.sub(r'ཧེ་ཧེ་(ཧེ་)+', r'ཧེ་ཧེ་ཧེ་', text)
    text = re.sub(r'ཧ་ཧ་(ཧ་)+', r'ཧ་ཧ་ཧ་', text)
    text = re.sub(r'ཧོ་ཧོ་(ཧོ་)+', r'ཧོ་ཧོ་ཧོ་', text)
    text = re.sub(r'ཨོ་ཨོ་(ཨོ་)+', r'ཨོ་ཨོ་ཨོ་', text)

    # Remove specific punctuation marks and special characters
    chars_to_ignore_regex = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�\/\{\}\(\)༽》༼《༄༅༈༑༠'|·×༆༸༾ཿ྄྅྆྇ྋ࿒ᨵ​’„╗᩺╚༿᫥ྂ༊ྈ༁༂༃༇༈༉༒༷༺༻࿐࿑࿓࿔࿙࿚༴࿊]"
    text = re.sub(chars_to_ignore_regex, '', text) + " "
    
    return text



def _load_json_with_comments(path):
    """Load JSON that may include // line comments (used by the shared var file)."""
    with open(path, 'r') as f:
        text = f.read()
    # [Reason] Allow department comments like // AB above each block in var
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    return json.loads(text)


def _project_var_path():
    """Path to stt-split-audio/var (shared FROM_ID/TO_ID by department)."""
    import os
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'var')


def _var_key_for_config(config_path):
    """Map config filename to a var key, e.g. ab_config_etext.json -> AB."""
    import os
    stem = os.path.splitext(os.path.basename(config_path))[0]
    for suffix in ('_config_etext', '_config', '_etext'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.upper()


def load_config_from_file(config_path):
    """Load configuration from the given JSON file.

    Supports optional EXTENDS to inherit from another JSON file (path is
    relative to the config). Child keys override the parent.

    Also injects FROM_ID / TO_ID from the shared ../var file, keyed by
    department/config name (see comments in var).
    """
    import os

    with open(config_path, 'r') as f:
        config = json.load(f)

    # [Reason] Let configs inherit shared keys from a parent JSON when needed
    extends = config.pop('EXTENDS', None)
    if extends:
        parent_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), extends)
        parent = load_config_from_file(parent_path)
        parent.update(config)
        config = parent

    # [Reason] Single place to edit ID ranges for every json_config
    var_path = _project_var_path()
    if os.path.isfile(var_path) and os.path.abspath(config_path) != os.path.abspath(var_path):
        shared = _load_json_with_comments(var_path)
        key = _var_key_for_config(config_path)
        section = shared.get(key)
        if isinstance(section, dict):
            for id_key in ('FROM_ID', 'TO_ID'):
                if id_key in section:
                    config[id_key] = section[id_key]

    return config


def parse_args_and_load_config():
    """Parse command-line arguments and load the configuration file."""
    parser = argparse.ArgumentParser(description="Run the audio processing pipeline with config.")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file.')
    
    args = parser.parse_args()

    # Load configuration from file
    return load_config_from_file(args.config)


def get_time_span(filename):
    """
    Extracts the time span in seconds from a filename.
    
    Args:
        filename (str): The filename from which to extract the time span.
    
    Returns:
        float: The time span in seconds, or 0 if extraction fails.
    """
    filename = filename.lower().replace(".wav", "").replace(".mp3", "")
    try:
        if "_to_" in filename:
            start, end = filename.split("_to_")
        else:
            start, end = filename.split("-")
        start = float(start.split("_")[-1])
        end = float(end.split("_")[0])
        return (end - start) / 1000 if "_to_" in filename else abs(end - start)
    except Exception as err:
        print(f"Error parsing filename '{filename}': {err}")
        return 0