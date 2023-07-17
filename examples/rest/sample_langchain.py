from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

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

def llm_chain_example():
    template = """
    {history}
    USER: {human_input}
    ASSISTANT:"""

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    llm_chain = LLMChain(
        llm=ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()]),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(human_prefix="USER", ai_prefix="ASSISTANT")
    )

    output = llm_chain.predict(human_input="Write a short poem about Pittsburgh.")
    output = llm_chain.predict(human_input="What does it mean?")

def load_qa_chain_example():
    loader = TextLoader('linux.txt')
    documents = loader.load()
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff", verbose=False)
    query = "When was Linux released?"
    print(f"{color.BOLD}Query:{color.END} {color.BLUE} {query}{color.END}")
    print(f"{color.BOLD}Response:{color.END} {color.GREEN}{chain.run(input_documents=documents, question=query)}{color.END}")

def retrieval_qa_example():
    prompt_template = """Use only the following pieces of context to answer the question at the end. Don't use any other knowledge.

    {context}

    USER: {question}
    ASSISTANT:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    loader = TextLoader('state_of_the_union.txt')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    db = Chroma.from_documents(documents=texts, embedding=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    questions = [
        "What is the American Rescue Plan?",
        "What did the president say about Ketanji Brown Jackson?",
        "Who is mentioned in the speech?",
        "To whom is the speech addressed?",
        "Tell me more about the Made in America campaign."
    ]

    for qn in questions:
        print(f"{color.BOLD}QUESTION:{color.END} {qn}")
        res = qa({'query': qn})
        print(f"{color.BOLD}RESPONSE:{color.END} {color.GREEN}{res['result']}{color.END}")
        print(f"{color.BOLD}SOURCE:{color.END} {color.BLUE}{repr(res['source_documents'][0].page_content)}{color.END}")
        print()

# Uncomment the demo you want to run below:

# llm_chain_example()
# load_qa_chain_example()
# retrieval_qa_example()
