import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool

from langchain.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv
import os
load_dotenv()



groq_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_key)

db = SQLDatabase.from_uri("sqlite:///chinook.db")

db.get_usable_table_names()

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

# agent_executor.run("Give me contact number of the customer named Mark from edmonton")

# # #Defining the LLM

# # #Defining the tool
# # tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_key)

# # #LLM prompt

# # prompt = ChatPromptTemplate.from_messages([("system","You are a powerfull Assistant"),
# #                                            ("human","{input}"),
# #                                            ("placeholder","{agent_scratchpad}")])



# # @tool
# # def word_count(word):
# #     """
# #     This is function takes a string and returns the length of it
# #     """
# #     print(word)
# #     return len(word)

# # tools = [tavily_tool, word_count]

# # search_agent = create_tool_calling_agent(llm,tools,prompt)

# # search_agent_executor = AgentExecutor(agent=search_agent, tools=tools)


def ask(prompt):
    response = agent_executor.run(prompt)
    return response
        

# # st.write("Search Agent")
st.title("Database Agent")

#Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# User Input handling and adding to the history
if prompt := st.chat_input():
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    with st.spinner('Generating...'):
        try:
            chat_history = "\n".join([f"{history['role']}:{history['content']}" for history in st.session_state.messages])
            chat_history += prompt
            # print("\n".join([f"{data['role']}:{data['content']}" for data in st.session_state.messages]))
            response = ask(chat_history)
        except Exception as e:
            response = "Please, try again later!"
    # st.success("Done!")
    # st.success('Done!')
    # Assistant chat write
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})