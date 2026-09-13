from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create the template
template = """You are a helpful assistant that translates {input_language} to {output_language}.

Text: {text}

Translation:"""

prompt = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template=template
)

# Initialize the LLM with API key
# Make sure you have GOOGLE_API_KEY in your .env file
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create a chain: template -> LLM
chain = prompt | model

# Run the chain (this will actually translate!)
response = chain.invoke({
    "input_language": "English",
    "output_language": "Spanish",
    "text": "Hello, how are you?"
})

print(response.content)
# Output: "Hola, ¿cómo estás?"
