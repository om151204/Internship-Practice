from pdf_loader import load_pdf
from agent import create_agent


def run(pdf_path):
    text = load_pdf(pdf_path)
    agent = create_agent()

    query = f"""
    Follow this sequence strictly:
    1. requirements_tool
    2. user_story_tool
    3. task_tool

    input:
    {text}

    Final Output must include:
    - Requirements
    - User Stories
    - Tasks 

    """
    result = agent.invoke(query)
    return result

if __name__ == "__main__":
    output = run("SRS.pdf")
    print("\nFINAL OUTPUT:- ")
    print(output)
