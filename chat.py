import os
import dspy
from dotenv import load_dotenv

# Load variables from the .env file into environment variables
load_dotenv()

# Configure DSPy with Groq (DSPy/LiteLLM automatically reads GROQ_API_KEY from the environment)
lm = dspy.LM("groq/llama-3.1-70b-versatile")
dspy.configure(lm=lm)


class QA(dspy.Signature):
  question: str = dspy.InputField()
  history: dspy.History = dspy.InputField()
  answer: str = dspy.OutputField()


predict = dspy.Predict(QA)
history = dspy.History(messages=[])

while True:
  question = input("Type your question, end conversation by typing finish: ")
  if question == "finish":
    break
  outputs = predict(question=question, history=history)
  print(f"\n{outputs.answer}\n")
  history.messages.append({"question": question, **outputs})

dspy.inspect_history()
