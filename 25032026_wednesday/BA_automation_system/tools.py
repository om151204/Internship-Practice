from langchain.tools import tool
from config import llm
from text_splitter import split_text

def load_requirement_prompt():
    try:
        with open("prompts/requirements.md") as f:
            return f.read()
    except FileNotFoundError as e:
        print("Please check the file path and try again",e)
    except Exception as e:
        print("Error",e)

@tool
def requirements_tool(input_text:str)->str:
    """
    Extract requirements from input text and return a string with only the requirements
    :param input_text: input text data
    :return: string of requirements
    """
    try:
        prompt_template = load_requirement_prompt()
        chunks = split_text(input_text)

        results = []

        for chunk in chunks:
            prompt = prompt_template + f"\n\n## Input\n{chunk}"
            response = llm.invoke(prompt).content
            results.append(response)
        return "\n".join(results)
    except Exception as e:
        print("Error in requirements_tool",e)
        return ""


def load_user_story_prompt():
    try:
        with open('prompts/user_story.md') as f:
            return f.read()
    except FileNotFoundError as e:
        print("Please check the file path and try again",e)
    except Exception as e:
        print("Error",e)

@tool
def user_story_tool(requirements:str):
    """
    Generate user stories
    :param requirements: requirement captured from tool1
    :return: User Stories generated from requirements
    """
    try:

        prompt_template = load_user_story_prompt()
        prompt = prompt_template + f"\n\n## Input\n{requirements}"

        return llm.invoke(prompt).content
    except Exception as e:
        print("Error in user_story_tool",e)

def load_task_prompt():
    try:
        with open("prompts/tasks.md","r") as f:
            return f.read()
    except FileNotFoundError as e:
        print("Please check the file path and try again",e)
    except Exception as e:
        print("Error",e)

@tool
def task_tool(user_stories:str):
    """Generate tasks"""
    try:
        prompt_template = load_task_prompt()
        prompt = prompt_template + f"\n\n### {user_stories}"
        return llm.invoke(prompt)
    except Exception as e:
        print("Error in task_tool",e)
