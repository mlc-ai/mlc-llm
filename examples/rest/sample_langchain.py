from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# First set the following in your environment:
# export OPENAI_API_BASE=http://127.0.0.1:8000/v1
# export OPENAI_API_KEY=EMPTY

# Note that Langchain does not currently support Pydantic v2:
# https://github.com/langchain-ai/langchain/issues/6841
# Please ensure that your `pydantic` version is < 2.0

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

llm_chain_example()
load_qa_chain_example()
