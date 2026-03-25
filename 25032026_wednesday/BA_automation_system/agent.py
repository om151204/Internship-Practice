from langchain_classic.agents import initialize_agent, AgentType
from config import llm
from tools import requirements_tool,user_story_tool,task_tool

def create_agent():
    """
    Create an agent on three tools and return it
    :return: None
    """
    try:
        tools = [requirements_tool,user_story_tool,task_tool]

        agent = initialize_agent(
            tools = tools,
            llm = llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose = True
        )

        return agent
    except Exception as e:
        print("Error in creating agent",e)

