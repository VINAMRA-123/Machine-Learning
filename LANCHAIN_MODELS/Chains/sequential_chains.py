from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
prompt1 = PromptTemplate(
    template="give me detail about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= "give 5 sentene summary of it \n{text}"
    ,input_variables=['text']
)

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()
chain  = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke(  {"topic":"marriage"})
print(result)
