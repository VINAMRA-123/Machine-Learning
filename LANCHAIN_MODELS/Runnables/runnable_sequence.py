from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template="Write a jokee on {topic}",
    input_variables= 'topic'
)
prompt2 = PromptTemplate(
    template= "Explain the following joke {text}"
)

parser = StrOutputParser()
chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke(input="Technology")
print(result)


