from typing import TypedDict, Optional
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
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
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert critical thinker. Review the provided research findings. Identify any conflicting information or redundancy, and provide a consolidated, accurate analysis."),
            ("human", "Here is research on {topic}: {finding}")
        ]
    )
    chain = template | llm
    response = chain.invoke({"topic": topic, "finding": search_result})
    return {'analysis': response.content}


def writer_node(state: AgentState):
    topic = state['topic']
    analysis = state['analysis']

    writer_template = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                "You are a world-class writer with infinite versatility. You are an expert content writer. Your goal is to transform the provided analysis into an engaging, well-structured article. \n\n"
                "Guidelines:\n"
                "1. First, analyze the topic to determine the ideal tone, style, and audience (e.g., adventurous for travel, precise for science, empathetic for advice).\n"
                "2. Adopt that persona completely.\n"
                "3. Tone: Professional yet accessible\n"
                "4. Write an engaging, well-structured article using Markdown with a clear Title, Introduction, and Subheadings.\n"
                "5. Length: Aim for 300-500 words.\n"
                "6. Strictly base your content on the provided Source Material. Do NOT add new facts."
            ),
            ("human", "Here is analysis on {topic}: {analysis}")
        ]
    )

    prompt_chain = writer_template | llm
    response = prompt_chain.invoke({"topic": topic, "analysis": analysis})
    return {'script': response.content}


workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("critical_thinker", critical_thinker_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critical_thinker")
workflow.add_edge("critical_thinker", "writer")
workflow.add_edge("writer", END)
app = workflow.compile()

if __name__ == "__main__":
    topic = "history of tea"
    app.invoke({"topic": topic})
    final_state = app.invoke({"topic": topic})
    
    print("\n" + "="*50)
    print("FINAL SCRIPT")
    print("="*50 + "\n")
    print(final_state['script'])