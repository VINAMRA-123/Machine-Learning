from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace
from langchain_core.runnables import RunnableParallel

prompt1 = PromptTemplate(
    template="give me notes about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= "give me quiz about {topic}",
    input_variables= ['topic']
)

prompt3 = PromptTemplate(
    template="merge the quiz and notes of {notes} and {quiz}",
    input_variables= ['quiz','notes']
)

import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=100,
    temperature=0.7,
)

model1 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
model2 = ChatHuggingFace(llm = llm)

parser = StrOutputParser()
parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 |parser
})
chain  = parallel_chain | prompt3 | model1 | parser
result = chain.invoke(  {"topic":"machine learning"})
print(result)
