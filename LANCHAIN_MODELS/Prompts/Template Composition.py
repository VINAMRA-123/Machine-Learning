# Combine multiple templates for complex prompts. This pattern allows you to build modular, reusable prompt components that can be mixed and matched for different use cases:
from langchain.prompts import PromptTemplate

# Define reusable template components
persona_template = "You are a {expertise} expert with {years} years of experience."
task_template = "Your task is to {action} the following {content_type}:"
format_template = "Please format your response as {format}."

# Combine templates
full_template = f"""
{persona_template}

{task_template}

{format_template}

Content: {{content}}
"""

# Create the prompt
prompt = PromptTemplate(
    input_variables=["expertise", "years", "action", "content_type", "format", "content"],
    template=full_template
)

# Use the composed template
result = prompt.format(
    expertise="Python",
    years="10",
    action="optimize",
    content_type="code",
    format="a list of improvements with explanations",
    content="def calculate_sum(numbers): total = 0; for n in numbers: total = total + n; return total"
)

print(result)
# 🧩 Why Use Template Composition?

#     • Modularity: Create reusable building blocks for different prompt parts
#     • Consistency: Ensure uniform structure across all prompts in your application
#     • Flexibility: Mix and match components for different scenarios
#     • Maintainability: Update one component to affect all prompts using it

# 💡 Real-World Example:

# Imagine building a customer support system where you need different combinations:

#     • Technical Support: persona_template + technical_task + detailed_format
#     • Sales Inquiry: persona_template + sales_task + friendly_format
#     • Complaint Handling: persona_template + empathy_task + solution_format

Same persona component, different task and format components!
