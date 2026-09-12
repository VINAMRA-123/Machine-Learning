from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-3.6-flash")
res  = llm.invoke("ai engineer roadmap")
print(res)