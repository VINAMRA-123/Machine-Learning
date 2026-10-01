from langchain.prompts import PromptTemplate

# Create a simple template
template = """You are a helpful assistant that translates {input_language} to {output_language}.

Text: {text}

Translation:"""

# Create the prompt template
prompt = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template=template
)

# Use the template
formatted_prompt = prompt.format(
    input_language="English",
    output_language="Spanish",
    text="Hello, how are you?"
)

print(formatted_prompt)

# Output will be:
# You are a helpful assistant that translates English to Spanish.
# 
# Text: Hello, how are you?
# 
# Translation:
