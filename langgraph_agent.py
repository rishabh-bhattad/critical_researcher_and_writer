from typing import TypedDict, Optional
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro'
)

class AgentState(TypedDict):
    topic: Optional[str]
    finding: Optional[str]
    analysis: Optional[str]
    script: Optional[str]

    
def research_node(state: AgentState):
    search_tool = DuckDuckGoSearchRun()
    topic = state['topic']
    search_result = search_tool.invoke(topic)
    return {'finding': search_result}


def critical_thinker_node(state: AgentState):
    topic = state['topic']
    search_result = state['finding']
    
