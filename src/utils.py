"""
utils.py -The Ingestion Pipeline for The Venture Analyst Agent.

Handles PDF parsing, text cleaning, and chunking.
This is the FIRST step in the pipeline: Raw PDF ->Clean Text ->Chunks.
"""
from tkinter.filedialog import test
import pdfplumber
import re
import os
from typing import List
def extract_text_from_pdf(pdf_path:str) ->str:
    """
    Extracts raw text from a pdf file,page by page.
    Args:
        pdf_path: Path to the pdf file
    Returns:
        A single string with all text,annoted with page numbers.
    Raises:
        FileNotFundError: If the pdf path doesn't exist.
        Exception: If the pdf cannot be parsed.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:    
        for i, page in enumerate(pdf.pages,start=1):
            page_text = page.extract_text()
            if page_text:
                full_text += f"\n\n[Page {i+1}]\n{page_text}"
    if not full_text.strip():
        raise ValueError("PDF appears to be empty or image-based(no extractable text).")
    return clean_text(full_text)
def extract_text_from_upload(uploaded_file) -> str:
    """
    Extracts text from a Streamlit UploadedFile object.
    This is the version we'll use in the Streamlit app, since
    Args:
        uploaded_file : A Streamit UploadedFile object.
    Returns:
            Cleaned text string from the pdf.    
    """
    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:    
        for i, page in enumerate(pdf.pages,start=1):
            page_text = page.extract_text()
            if page_text:
                full_text += f"\n\n[Page {i+1}]\n{page_text}"
    if not full_text.strip():
        raise ValueError("PDF appears to be empty or image-based(no extractable text).")
    return clean_text(full_text)
def clean_text(text:str) -> str:
    """
    Cleans extracted pdf text by removing artifacts.
    Common Pdf issues:
    - Excessive whitespace
    - Line breaks in the middle of sentences
    - Special characters from encoding issues
    Args :
        text: Raw text extracted from the pdf.
    Returns:
        Cleaned text string.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}',' ',text)
    text = re.sub(r'\x00','',text)
    text=text.strip()
    return text
def chunk_text(text:str,chunk_size:int=1000,overlap:int=200) -> List[str]:
    """
    Splits text into overlapping Chunks for the vector database.
    why overlap? Because a claim like "we have 50% market share"
    might be split right in the middle without overlap.
    Args:
        text: The full cleaned text from the pdf.
        chunk_size: Maximum charcters per chunk.
        overlap: number of overlapping characters between chunks.
    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end <len(text):
            break_point=[
                text.rfind('.',start, end),
                text.rfind('\n',start,end),
                text.rfind(' ',start,end)
            ]
            best_break = max(break_point)
            if best_break> start:
                end = best_break +1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)    
        start = end - overlap if end < len(text) else end
    return chunks

 