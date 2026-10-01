from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template="Write a jokee on {topic}",
    input_variables= 'topic'
)
# prompt2 = PromptTemplate(
#     template= "Explain the following joke {text}"
# )

parser = StrOutputParser()
chain1 = RunnableSequence(prompt1,model,parser)
def word_count(text):
    num = len(text.split(" "))
    return num
        
    
parallel =  RunnableParallel({
    'joke':RunnablePassthrough(),
    'words' : RunnableLambda(word_count)
})
ans = RunnableSequence(chain1,parallel)
result = ans.invoke(input="Technology")
print(result)


