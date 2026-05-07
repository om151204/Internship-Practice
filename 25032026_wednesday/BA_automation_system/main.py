from langchain_core.messages import HumanMessage

from pdf_loader import load_pdf
from agent import agent_creation


def run(pdf_path):
    try:
        text = load_pdf(pdf_path)
        agent = agent_creation()
        # The prompt content
        query_content = f"""
        Follow this sequence strictly:
        1. requirements_tool
        2. user_story_tool
        3. task_tool
    
        input:
        {text[:200]}
    
        Final Output must include:
        - Requirements
        - User Stories
        - Tasks 
        """


        inputs = {"messages": [HumanMessage(content= query_content)]}

        result = agent.invoke(inputs)

        # LangGraph returns the full state; we extract the last message (the AI's answer)
        return result["messages"][-1].content
    except ConnectionError as e:
        return f"Error: {e}"


if __name__ == "__main__":
    output = run("SRS.pdf")
    print("\nFINAL OUTPUT:- ")
    print(output)
