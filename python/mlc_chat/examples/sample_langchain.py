from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# First set the following in your environment:
# export OPENAI_API_BASE=http://127.0.0.1:8000/v1
# export OPENAI_API_KEY=EMPTY

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
