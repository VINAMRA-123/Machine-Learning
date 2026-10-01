from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a chat template with system and human messages
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} that helps with {task}."),
    ("human", "{user_input}"),
])

# Format the template
messages = chat_template.format_messages(
    role="programming tutor",
    task="Python coding",
    user_input="How do I read a CSV file?"
)

# Use with your model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")  # Set API key explicitly
)
response = llm.invoke(messages)
print(response.content)

"""📋 Understanding Message Roles:

    • "system": Sets the AI's behavior, personality, or constraints. Like giving instructions to an assistant
    • "human": The user's input or question
    • "assistant": The AI's previous responses (used for conversation history)

✨ Benefits of ChatPromptTemplate:

    • Automatically formats messages in the correct structure for chat models
    • Supports dynamic variables in any message type
    • Makes it easy to build conversational applications
    • Maintains consistency across different chat model providers"""
