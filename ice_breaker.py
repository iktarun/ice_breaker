import os
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI

# I have tried it out with the Azure
if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain")
    print(os.environ.get("AZURE_OPENAI_API_KEY"))

    information = """
        Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor...
        (Truncated for brevity in this example.)
    """

    summary_template = """
    Given the information {information} about a person, create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = AzureChatOpenAI(
        deployment_name="gpt-4",  # Replace with your deployment name
        temperature=0,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-12-01-preview",
    )

    chain = LLMChain(prompt=summary_prompt_template, llm=llm)
    res = chain.run(information=information)

    print(res)
