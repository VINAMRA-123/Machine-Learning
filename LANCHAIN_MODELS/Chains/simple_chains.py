from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

Prompts = PromptTemplate(
    template='Give me name of 5 facts about {pornstars}',
    input_variables= ['pornstars']
)

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()

chain = Prompts | model | parser

result  = chain.invoke(input="cricket")

print(result)
print(chain.get_graph().print_ascii)