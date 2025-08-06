from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


class SimpleStrOutputParser:
    def parse(self, output):
        # Just return the output as-is
        return output

    def __call__(self, output):
        return self.parse(output)

# 2. Define the prompt template
prompt = PromptTemplate.from_template(
    "What is the capital of {topic}?"
)

# 3. Define the model
model = ChatOllama(model="llama2")  # Using Ollama

# 4. Chain the components together using LCEL syntax
chain = (
    {"topic": RunnablePassthrough()}  # Accept user input
    | prompt                          # Transform it into a prompt message
    | model                          # Call the model
    | SimpleStrOutputParser()              # Parse the output as a string
)

# 5. Execute the chain with input "Germany"
result = chain.invoke("China")

print("User prompt: 'What is the capital of China?'")
print("Model answer:", result)

print("\nParsed fields:")
print("content:", repr(result.content))
print("additional_kwargs:", result.additional_kwargs)
print("response_metadata:", result.response_metadata)
print("id:", getattr(result, "id", None))
print("usage_metadata:", getattr(result, "usage_metadata", None))
