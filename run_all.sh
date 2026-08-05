#!/bin/bash

# Run yt_download.py
echo "Running yt_download.py..."
python ../audio_download_and_split/yt_download.py --config ../json_config/ab_config.json
if [ $? -ne 0 ]; then
  echo "Error running yt_download.py. Exiting..."
  exit 1
fi

# Run run_inference.py
echo "Running run_inference.py..."
python ../inference_runner/run_inference_text.py --config ../json_config/ab_config.json
if [ $? -ne 0 ]; then
  echo "Error running run_inference.py. Exiting..."
  exit 1
fi

# Run make_csv.py
echo "Running make_csv.py..."
python ../make_db_csv/make_csv.py --config ../json_config/ab_config.json
if [ $? -ne 0 ]; then
  echo "Error running make_csv.py. Exiting..."
  exit 1
fi

# [Reason] AB etext transfer lives in download_doc.py (download + transfer + upload CSV);
# transfer_text.py is obsolete here and pointed at the wrong etexts path.
echo "Running download_doc.py (AB etext download + text transfer)..."
python ../make_db_csv/download_doc.py --config ../json_config/ab_config_etext.json
if [ $? -ne 0 ]; then
  echo "Error running download_doc.py. Exiting..."
  exit 1
fi

echo "All scripts ran successfully!"
