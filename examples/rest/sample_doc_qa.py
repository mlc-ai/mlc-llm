from docutils.core import publish_file
import rst2txt
import glob
import os

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredRSTLoader, DirectoryLoader

# First set the following in your environment:
# export OPENAI_API_BASE=http://127.0.0.1:8000/v1
# export OPENAI_API_KEY=EMPTY

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

prompt_template = """Use only the following pieces of context to answer the question at the end. Don't use any other knowledge.

{context}

USER: {question}
ASSISTANT:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

loader = DirectoryLoader("../../docs", glob='*/*.rst', show_progress=True, loader_cls=UnstructuredRSTLoader, loader_kwargs={"mode": "single"})
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
db = Chroma.from_documents(collection_name="abc", documents=texts, embedding=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True, 
    chain_type_kwargs={"prompt": PROMPT}
)
while True:
    qn = input(f"{color.BOLD}QUESTION:{color.END} ")
    res = qa({'query': qn})
    print(f"{color.BOLD}RESPONSE:{color.END} {color.GREEN}{res['result']}{color.END}")
    print(f"{color.BOLD}SOURCE:{color.END} {color.BLUE}{repr(res['source_documents'][0].page_content)}{color.END}")
    print()