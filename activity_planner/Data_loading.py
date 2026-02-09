
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

import json
from langchain_core.documents import Document
from glob import glob
from langchain_community.document_loaders import JSONLoader
# If wiki_00 is a JSON Lines file (one JSON object per line)


def process_json_files(directory_path):
    dir_paths = glob(directory_path + '/*')
    documents = []
    jq_schema = """
    {
    id: .id,
    title: .title,
    text: (
        [
        (if .intro then "INTRO:\\n" + .intro else null end),
        (if .understand then "UNDERSTAND:\\n" + .understand else null end),
        (if .get_in then "GET IN:\\n" + .get_in else null end),
        (if .get_around then "GET AROUND:\\n" + .get_around else null end),
        (if .see then "SEE:\\n" + .see else null end),
        (if .eat then "EAT:\\n" + .eat else null end),
        (if .drink then "DRINK:\\n" + .drink else null end),
        (if .stay_safe then "STAY SAFE:\\n" + .stay_safe else null end),
        (if .go_next then "GO NEXT:\\n" + .go_next else null end)
        ]
        | map(select(. != null))
        | join("\\n\\n")
    )
    }
    """

    for dir_path in dir_paths:
        files = glob(dir_path + '/*')
        for file in files:
        
            loader = JSONLoader(
                file_path=file,
                jq_schema=jq_schema, # Replace '.text' with the key containing your data
                text_content=False,
                json_lines=True 
            )
            docs = loader.load()
            documents.extend(docs)
    return documents
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    )
    return text_splitter.split_documents(documents)
