from crewai import Agent
from travelTools import search_web_tool
#from TravelTools import search_web_tool, web_search_tool
from crewai import LLM
from langchain_openai import ChatOpenAI


# Initialize LLM
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)


# Agents
guide_expert = Agent(
    role="City Local Guide Expert",
    goal="Provides information on things to do in the city based on user interests.",
    backstory="A local expert passionate about sharing city experiences.",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=LLM(model="ollama/llama3.2",base_url="http://localhost:11434"),
    allow_delegation=False,
)

location_expert = Agent(
    role="Travel Trip Expert",
    goal="Provides travel logistics and essential information.",
    backstory="A seasoned traveler who knows everything about different cities.",
    tools=[search_web_tool],  
    verbose=True,
    max_iter=5,
    llm= LLM(model="ollama/llama3.2",base_url="http://localhost:11434"),   # ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    allow_delegation=False,
)

planner_expert = Agent(
    role="Travel Planning Expert",
    goal="Compiles all gathered information to create a travel plan.",
    backstory="An expert in planning seamless travel itineraries.",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=LLM(model="ollama/llama3.2",base_url="http://localhost:11434"),
    allow_delegation=False,
)