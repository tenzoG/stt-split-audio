from docx import Document
from pathlib import Path
import os
import pandas as pd
from fast_antx.core import transfer
import numpy as np
import gdown
from docx import Document

def docx_to_txt(docx_path):
        doc = Document(docx_path)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)


def download_etext(gd_url,file_name):
    if not os.path.exists('etexts'):
        os.makedirs('etexts')

    if not os.path.exists('docx'):
        os.makedirs('docx')
    
    if os.path.exists(f'etexts/{file_name}.txt'):
        return
    docx_url, _ = os.path.split(gd_url)
    docx_url = os.path.join(docx_url, 'export?format=docx')
    # [Reason] gdown>=5 removed fuzzy; export URL already targets the docx download
    docx_path = gdown.download(docx_url, output=f'docx/{file_name}.docx', quiet=False)
    # Convert the .docx file to text
    text = docx_to_txt(docx_path)
    # Create a .txt path with the same name as the .docx file
    txt_path = os.path.join('etexts/', file_name + '.txt')
    # Save the text to a .txt file
    with open(txt_path, 'w',encoding='utf-8') as f:
        f.write(text.replace('\n', ' '))


def extract_tsv_text(tsvFile, ColumnNumber):
    """extracts text from dataframe using column number
    Args:
        tsvFile (Dataframe): dataframe of predicted tsv file
        ColumnNumber (integer/string):column name of the text to be extracted

    Returns:
        string: extracted text from tsv file
    """
    # read the tsv file
    predictedText = tsvFile[ColumnNumber].tolist()
    # to avoid unwanted splits in a word we replace space with _
    for count, text in enumerate(predictedText):
        predictedText[count] = predictedText[count].replace(" ", "_")
    predictedText = "\n".join(" ".join(predictedText).split())
    print("extracted text from tsv file..")
    return predictedText


def get_original_text(OriginalText):
    """reads the original text and removes unwanted characters

    Args:
        OriginalText (string): location of the original text file

    Returns:
        string: original text without unwanted characters
    """
    target = Path(f"{OriginalText}").read_text(encoding="utf-8")
    # remove unwanted characters
    target = target.replace("“", "").replace("”", "")
    target = target.replace("\n","")

    print("extracted text from original file..")

    return target

def transfer_text(OriginalText, PredictedTSV, file_name, ColumnNumber='inference_transcript'):
    """transfers the annotation from predicted text to original text and returns a dataFrame

    Args:
        OriginalText (string): location of the original string
        PredictedTSV (string): location of the predicted tsv file
        ColumnNumber (int/string): name of the column in which transcribed text is there in .tsv file

    Returns:
        dataFrame: dataFrame that contains transferred annotation on original text
    """
    tsvFile = pd.read_csv(f"{PredictedTSV}", sep="\t")
    # [Reason] Match full ID prefix (AB IDs are >11 chars; old str[0:11] filter returned empty frames)
    tsvFile = tsvFile[
        tsvFile['file_name'].str.startswith(f"{file_name}_", na=False)
        | (tsvFile['file_name'] == file_name)
    ].copy()

    if tsvFile.empty:
        print(f"⚠ No rows found for {file_name}")
        return tsvFile, 'No data'

    tsvFile.sort_values(by=['file_name'], inplace=True)

    source = extract_tsv_text(tsvFile, ColumnNumber)
    target = get_original_text(OriginalText)
    annotation = [["segment", "(\n)"]]
    transferredText = transfer(source, annotation, target).split("\n")
    original_length = len(tsvFile)
    transferred_length = len(transferredText)
    if transferred_length > original_length:
        # [Reason] Compute truncate count before slicing so status is not always Truncated 0
        status = f'Truncated {transferred_length - original_length}'
        transferredText = transferredText[:original_length]
        tsvFile[ColumnNumber] = transferredText
    elif transferred_length < original_length:
        transferredText = transferredText + [np.nan]*(original_length - transferred_length)
        tsvFile[ColumnNumber] = transferredText
        status = f'Padded {original_length - transferred_length}'
    else:
        tsvFile[ColumnNumber] = transferredText
        status = 'Normal'

    # returns a dataFrame
    return tsvFile, status
