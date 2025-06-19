from crewai import Agent
from travelTools import search_web_tool
# from crewai import LLM
form langchain_groq import ChatGroq

llm= ChatGroq(
    model=""
)

# Agents

#Guide Expert
guide_expert=Agent(
    role="City Local Guide Expert",
    goal="Provides information on things to do in the city based on user intersts"
    backstory="A locla expert passionate about sharing city experiences."
    tools=[search_web_tool]
    verbose=True,
    llm=llm,
    allow_delegation=False,
)


