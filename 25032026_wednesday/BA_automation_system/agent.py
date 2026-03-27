from langchain.agents import create_agent
from config import llm
from tools import requirements_tool,user_story_tool,task_tool

def agent_creation():
    """
    Create an agent on three tools and return it
    :return: None
    """
    try:
        tools = [requirements_tool,user_story_tool,task_tool]

        agent = create_agent(
            model = llm,
            tools = tools,
            system_prompt = """
            You are a Business Analyst Agent.
        Workflow:
        1. Generate functional and non-functional requirements from the provided document.
        2. Generate detailed user stories based on the requirements.
        3. Generate implementation tasks from the user stories.
        
        Rules:
        - Use only the provided document content.
        - Do not invent missing business logic.
        - Always call tools in order: generate_requirements -> generate_user_stories -> generate_tasks
    """
        )
        return agent
    except Exception as e:
        print("Error in creating agent",e)



