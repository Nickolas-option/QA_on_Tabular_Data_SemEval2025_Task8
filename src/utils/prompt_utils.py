import time
import pandas as pd
from typing import List, Dict

# Placeholder imports - these will be resolved later
from ..data.loader import load_table
from .embedding_utils import get_relevant_rows_by_cosine_similarity 
# from ..clients.openai_client import OpenAIClient # Avoid circular dependency for type hint

def generate_prompt(row: dict, is_sample: bool) -> str:
    """
    Generate prompt for the LLM to generate Python code.
    """
    start_time = time.time()
    dataset = row["dataset"]
    question = row["question"]
    # Note: load_table is defined in src/data/loader.py
    df = load_table(dataset, is_sample)

    exp_prompt2 = f""""

1. You are two of the most esteemed Pandas DataScientists engaged in a heated and truth-seeking debate. You are presented with a dataframe and a question. Begin dialogue by rigorously discussing your reasoning step by step, ensuring to address all aspects of the checklist. In your discourse, meticulously articulate the variable type necessary to derive the answer and confirm that each column referenced is indeed present in the dataframe. Conclude your debate by providing the code to answer the question, ensuring that the variable result is explicitly assigned to the answer. Remember, all code must be presented in a single line, with statements separated by semicolors.
2. Refrain from importing any additional libraries beyond pandas and numpy.
3. The dataframe, df, is already populated with data for your analysis; do not initialize it, but focus solely on manipulating df to arrive at the answer.
4. If the question requires multiple entries, always utilize .tolist() to present the results.
5. If the question seeks a single entry, ensure that only one value is output, even if multiple entries meet the criteria.

You MUST FOLLOW THE CHECKLIST, ANSWER EACH OF ITS QUESTIONS (REASONING STEP), AND ONLY THEN OUTPUT THE FINAL ANSWER BASED ON THOSE ANSWERS:
1) How many values should be in the output?
2) Values (or one value) from which column (only one!) should the answer consist of?
3) What should be the type of value in the answer?

Example of a task:
Question: Identify the top 3 departments with the most employees.
<Columns> = ['department', 'employee_id']
<First_row> = ('department': 'HR', 'employee_id': 101)
Reasoning: Count the number of employees in each department, sort, and get the top 3. The result should be a list of department names.
Checklist:
1) The output should consist of 3 values.
2) The values should come from the 'department' column.
3) The type of value in the answer should be a list of strings.
Code: result = df['department'].value_counts().nlargest(3).index.tolist()

Your data to process:
<question> = {question}

- Make absolute sure that all columns used in query are present in the table.
<columns_in_the_table> = {[col for col in df.columns]}
<first_rows_of_table> = {df.head(3).to_string()}
YOUR Reasoning through dialogue and Code (Start final code part by "Code:"):

"""
    # print(f"Time for preparing generate_prompt data: {time.time() - start_time:.4f} seconds")
    return exp_prompt2

def create_reflection_prompt(question: str, df: pd.DataFrame, tracebacks: List[str]) -> str:
    """Creates a prompt for the LLM to reflect on failed solutions."""
    return (
        f"The following solutions failed for the task: \"{question}\"\n\n"
        + '\n'.join([f'Solution {i+1} Error:\n{traceback}\n' for i, traceback in enumerate(tracebacks)])
        + "\nDF info: \n"
        + f"<columns_to_use> = {str([(col, str(df[col].dtype)) for col in df.columns])}\n"
        + f"<first_row_of_table> = {str(df.head(1).to_dict(orient='records')[0])}\n"
        + "YOUR answer in a single line of pandas code:\n"
        + "Please craft a new solution considering these tracebacks. Output only fixed solution in one line:\n"
    )

def generate_voting_prompt(solutions: List[Dict], question: str, dataset_name: str, is_sample: bool):
    """Generates a prompt for the LLM to vote on the best solution."""
    # Note: load_table is defined in src/data/loader.py
    df = load_table(dataset_name, is_sample)

    voting_prompt = f"""
Examples of deducing answer types:
1. If the question is "Do we have respondents who have shifted their voting preference?" the answer type is **Boolean** because the response should be True/False.
2. If the question is "How many respondents participated in the survey?" the answer type is **Integer**
3. If the question is "List the respondents who preferred candidate X?" the answer type is **List** because the response requires a collection of values.
4. If the question is "What is the average age of respondents?" the answer type is **Number** because the response should be a decimal value.
5. If the question is "What is the name of the candidate with the highest votes?" the answer type is **String** because the response is a single textual value.

Given the following solutions and their results for the task: "{question}"

{'\n'.join([f'Solution Number {i+1}:\nCode: {r["code"]} Answer: {str(r["result"])[:50]} (may be truncated)\n' for i, r in enumerate(solutions)])}

Instructions:
- Deduce the most probable and logical result to answer the given question. Then output the number of the chosen answer.
- If you are presented with end-to-end solution, it should not be trusted for numerical questions, but it is okay for other questions.
- Make absolute sure that all columns used in solutions are present in the table. SQL query may use additional double quotes around column names, it's okay, always put them. Real Tables columns are: {df.columns}
- If the column name contain emoji or unicode character make sure to also include it in the column names in the query.
- If several solutions are correct, return the lowest number of the correct solution.
- Otherwise, return the solution number that is most likely correct.
- If the question ask for one entry, make sure to output only one, even if multiple qualify.

You should spell out your reasoning step by step and only then provide code to answer the question. In the reasoning state it is essentianl to spell out the answers' variable type that should be sufficient to answer the question. Also spell out that each column used is indeed presented in the table. The most important part in your reasoning should be dedicated to comparing answers(results) from models and deducing which result is the most likely to be correct, then choose the model having this answer.
First, predict the answer type for the question. Then give your answer which is just number of correct answer with predicted variable type.  Start reasoning part with "REASONING:" and final answer with "ANSWER:".
"""
    return voting_prompt

def get_sql_prompt(failed_solutions, question, column_names, df, ai_client):
    """Generates a prompt for the LLM to create a SQL query."""
    # Note: get_relevant_rows_by_cosine_similarity is defined in src/utils/embedding_utils.py
    # ai_client is an instance of OpenAIClient from src/clients/openai_client.py
    relevant_rows_md = "Error generating relevant rows" # Default fallback
    try:
        relevant_rows = get_relevant_rows_by_cosine_similarity(df, question, ai_client)
        relevant_rows_md = relevant_rows.head(3).to_markdown()
    except Exception as e:
        print(f"Error getting relevant rows for SQL prompt: {e}")

    return f"""
Some Python attempts failed with errors:
{', '.join([r["result"] for r in failed_solutions if isinstance(r.get('result'), str)])}

The task was: {question}

Here are some examples of SQL queries for similar tasks:
Example 1:
Task: Is there any entry where age is greater than 30?
REASONING:
1. Identify the column of interest, which is 'age'.
2. Determine the condition to check, which is 'age > 30'.
3. Use the SELECT statement to retrieve a boolean result indicating the presence of such entries.
4. Apply the WHERE clause to filter rows based on the condition 'age > 30'.
5. Use the EXISTS clause to ensure the query outputs 'True' if any row matches the condition, otherwise 'False'.
6. Ensure the table name is 'temp_table' and the column name is enclosed in double quotes to handle any spaces or special characters.
7. Verify that the query outputs 'True' or 'False' when presented with a yes or no question.
CODE: ```SELECT CASE WHEN EXISTS(SELECT 1 FROM temp_table WHERE "age" > 30) THEN 'True' ELSE 'False' END;```

Example 2:
Task: Count the number of entries with a salary above 50000.
REASONING:
1. Identify the column of interest, which is 'salary'.
2. Determine the condition to filter the data, which is 'salary > 50000'.
3. Use the SELECT COUNT(*) statement to count the number of rows that meet the condition.
4. Apply the WHERE clause to filter rows based on the condition 'salary > 50000'.
5. Ensure the table name is 'temp_table' and the column name is enclosed in double quotes to handle any spaces or special characters.
CODE: ```SELECT COUNT(*) FROM temp_table WHERE "salary" > 50000;```

Write a correct fault-proof SQL SELECT query that solves this precise task.
Rules:
- Your SQL query should be simple with just SELECT statement, without WITH clauses.
- Your SQL query should output the answer, without a need to make any intermediate calculations after its finish
- Use only basic SQL operations from SQLAlchemy (SELECT, FROM, WHERE, GROUP BY, etc.)
- Make sure not to use "TOP" operation as it is not presented in SQLite
- If present with YES or NO question, Query MUST return 'True' or 'False'
- Write pure SQL only without any quotes around the query
- If the question asks about several values, your query should return a list
- If the question ask for one entry, make sure to output only one, even if multiple qualify.
- Equip each column name into double quotes
- Equip each string literal into double quotes
- Use COALESCE( ..., 0) to answer with 0 if no rows are found and the question asks for the number of something.
- If it is Yes/No question, make sure that your query output only True or False.
- In the reasoning spell out that each column used is indeed presented in the table.
- Enclose your code into ```
- SELECT close MUST contain ONLY ONE column. For example, it must be only author's name, not name and id.
- Before writing code give extensive yet precise and specific reasoning for each step of your solution. Start reasoning part by "REASONING:" and code part by "CODE:"

Table name is 'temp_table'.
Available columns and types: {', '.join([f'"{col}": {str(df[col].dtype)}' for col in column_names])}

Top 3 rows with highest cosine similarity: {relevant_rows_md}
YOUR RESPONSE:

"""

def reformulate_question(dataset_name: str, question: str, is_sample: bool, ai_client):
    """
    Analyzes potential ambiguities in the question and reformulates it to be more precise using an LLM.
    Returns the reformulated question.
    """
    # Note: load_table is defined in src/data/loader.py
    # ai_client is an instance of OpenAIClient from src/clients/openai_client.py
    df = load_table(dataset_name, is_sample)
    schema = ', '.join([f"{col} ({df[col].dtype})" for col in df.columns])
    try:
        sample_data = df.head(5).to_markdown()
        sample_data = sample_data.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        print(f"Error encoding sample data to markdown: {e}")
        sample_data = "Error generating sample data markdown."

    # Ask about ambiguities
    ambiguities_prompt = f"""Given this dataset schema: {schema}
And this sample data (attention: it is just 5 first rows, not whole dataset):
{sample_data}

For this question: "{question}"

What are 3 potential severe ambiguities or unclear aspects that could prevent from answering this question correctly? Important: DO NOT PROVIDE YOUR OWN DEFINITIONS OR CONSTRAINTS, only use information from provided question and schema. ALWAYS CHECK IF RELEVANT COLUMN IS IN SCHEME AND INCLUDE IT INTO THE QUESTION. DONT ASK FOR UNIQUE WHERE IT IS NOT SPECIFIED EXPLICITLY."""

    try:
        ambiguities = ai_client.generate_response(ambiguities_prompt, model="meta-llama/llama-3.3-70b-instruct").strip()
    except Exception as e:
        print(f"Error generating ambiguities from LLM: {e}")
        ambiguities = "Could not generate ambiguities."

    # Ask for reformulation
    reformulation_prompt = f""""
examples:
Original request:
Are there any Pokémon with a total stat greater than 700?
Reformulated request:
(
"details": "1. The original question lacks column specificity. We need to reference the exact column names.
2. We should specify how Pokémon are uniquely identified in the dataset.
3. Need to be precise about the total stat column name.",
"result": Are there any unique Pokémon, identified by their `number`, in the dataset where the `total` value is greater than 700?
)

Original request:
How many posts are in Spanish?
Reformulated request:
(
"details": "1. The original question is vague about how Spanish language is identified.
2. Need to specify the exact column name and value to check.
3. Should clarify we want a count of matching posts.",
"result": What is the count of posts in the dataset where the 'lang' column value exactly matches 'es'?
)

Original request:
What is the average rating of the posts in Russia from 2015 to 2025 in Moscow?
Reformulated request:
(
"details": "1. The original question specifies a geographic location and time frame, which needs to be reflected in the dataset.
2. Need to clarify the exact column name for the rating.
3. Should mention if we want the average rating for all posts or a specific subset based on the provided criteria.",
"result": What is the average rating of posts in the dataset where the 'status' column value is 'published' and the 'date' column is between '2015-01-01' and '2025-12-31' in Moscow?
)

#### Task Description:
As the AI assistant, your task is to rewrite the NL entered by the user based on the given
database information and reflection.
This NL has some flaws and got bad generation in the downstream models, so you need to make this
NL as reliable as possible.
The rewritten NL should express more complete and accurate database information requirements
as far as possible. In order to do this task well, you need to follow these steps to think and
process step by step:
1. Please review the given reflection and DB information, and first check whether the NL contains
the corresponding key information and the corresponding flaws. If they exists, please modify,
supplement or rewrite it in the statement of NL by combining the reflection and DB.
2. Please rewrite the original NL based on the above process. On the premise of providing more
complete and more accurate database information, the structure of the rewritten NL should be similar
to the original statement as far as possible. All rewritten statements do not allow delimiters,
clauses, additional hints or explanations. DONT CONVERT IT INTO QUERY. DONT ADD UNIQUE WHERE IT IS NOT SPECIFIED EXPLICITLY. PREFER NAMES INSTEAD OF IDs when presenting the answer.
(
"details": <YOUR STEP-BY-STEP THINKING DETAILS>,
"result": <YOUR FINAL REWRITED NL>
)
### INPUT:
SCHEMA: # Fill the database content
{schema}
NL: # Fill the flaw NL
{question}
Possible Ambiguities:
{ambiguities}

### OUTPUT:

    """
    try:
        reformulated_response = ai_client.generate_response(reformulation_prompt, model="meta-llama/llama-3.3-70b-instruct").strip()
        # Extract the result part after the last occurrence of 'result:' (case-insensitive)
        match = re.search(r"result['"]?:\s*['"]?(.*?)['"]?\s*\)?\s*$", reformulated_response, re.IGNORECASE | re.DOTALL)
        if match:
            reformulated_question = match.group(1).strip().strip('"')
        else:
            # Fallback: Try finding the last line after 'result:'
            parts = re.split(r"result['"]?:", reformulated_response, flags=re.IGNORECASE)
            if len(parts) > 1:
                 reformulated_question = parts[-1].strip().split('\n')[0].strip().strip('"')
            else:
                 print("Warning: Could not parse reformulated question from LLM response. Using original question.")
                 reformulated_question = question # Fallback to original if parsing fails

    except Exception as e:
        print(f"Error generating reformulated question from LLM: {e}")
        reformulated_question = question # Fallback to original on error

    return reformulated_question 