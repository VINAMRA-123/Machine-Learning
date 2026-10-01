# Partial Templates
# Pre-fill some variables while leaving others dynamic. This is useful when some values are constant (like current date, system settings, or default values) while others change with each use:
from langchain.prompts import PromptTemplate
from datetime import datetime

# Template with multiple variables
template = """Date: {date}
User: {user_name}
Task: {task}

Please complete the following: {request}"""

# Create a partial template with date pre-filled
prompt = PromptTemplate(
    input_variables=["user_name", "task", "request"],
    template=template,
    partial_variables={"date": datetime.now().strftime("%Y-%m-%d")}
)

# Now you only need to provide the remaining variables
formatted = prompt.format(
    user_name="Alice",
    task="Code Review",
    request="Review this Python function for best practices"
)

print(formatted)
# 🔑 When to Use Partial Templates:

#     • Timestamps: Automatically include current date/time in logs or reports
#     • System Info: Pre-fill environment, version, or configuration details
#     • User Context: Set user preferences or settings once, reuse many times
#     • Default Values: Provide sensible defaults that can be overridden

# Pro Tip: Partial templates are perfect for creating specialized versions of general templates. For example, create a general email template, then use partials to create specific versions for support, sales, or notifications.
