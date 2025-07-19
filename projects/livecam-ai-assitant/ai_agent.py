from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


from dotenv import load_dotenv
from tools import anaylze_image_with_query

load_dotenv()

system_prompt = """You are Dora — a witty, clever, and helpful assistant"
    Here’s how you operate:
        - FIRST and FOREMOST, figure out from the query asked whether it requires a look via the webcam to be answered, if yes call the analyze_image_with_query tool for it and proceed.
        - Dont ask for permission to look through the webcam, or say that you need to call the tool to take a peek, call it straight away, ALWAYS call the required tools have access to take a picture.
        - When the user asks something which could only be answered by taking a photo, then call the analyze_image_with_query tool.
        - Always present the results (if they come from a tool) in a natural, witty, and human-sounding way — like Dora herself is speaking, not a machine.
    Your job is to make every interaction feel smart, snappy, and personable. Got it? Let’s charm your master!"
    """


llm =ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    temperature=0.7,
)

def ask_agent(user_query: str)-> str:
    agent =create_react_agent(
    model=llm,
    tools=[anaylze_image_with_query],
    prompt=system_prompt
)   
    input_messages= { "messages": [{"role": "user", "content": user_query}]}
    response = agent.invoke(input_messages)

    return response['messages'][-1].content


user_query="what colour tshirt i wear?, it is tshirt or shirt?"

answer= ask_agent(user_query)
print(answer) 