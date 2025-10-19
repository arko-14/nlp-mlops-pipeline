import argparse,os
from datasets import load_dataset

def main(src:str,out:str):
    os.makedirs(out,exist_ok=True)

    ds = load_dataset("csv", data_files=src)