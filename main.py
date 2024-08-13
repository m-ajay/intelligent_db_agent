import os
import pandas as pd

from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


openai_api_version = "2023-05-15"
azure_deployment = os.getenv("AZURE_DEPLOYMENT")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_version = "0613"

CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""


def run():
    df = pd.read_csv("./data/people-1000.csv").fillna(value=0)
    model = AzureChatOpenAI(
        openai_api_version=openai_api_version,
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        model_version=model_version,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    agent = create_pandas_dataframe_agent(
        llm=model, df=df, verbose=True, allow_dangerous_code=True
    )
    QUESTION = "Assume the current year is 2024, how many Engineers (regardeless of the industry) are females?"
    agent.invoke(CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX)


if __name__ == "__main__":
    run()
