import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from dotenv import load_dotenv
import os
load_dotenv()


groq_key = os.getenv('GROQ_API_KEY')
tavily_key = os.getenv('TAVILY_API_KEY')


#Defining the LLM
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_key)

#Defining the tool
tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_key)

#LLM prompt

prompt = ChatPromptTemplate.from_messages([("system","You are a powerfull Assistant"),
                                           ("human","{input}"),
                                           ("placeholder","{agent_scratchpad}")])

tools = [tavily_tool]



search_agent = create_tool_calling_agent(llm,tools,prompt)

search_agent_executor = AgentExecutor(agent=search_agent, tools=tools)


def ask(prompt):
    response = search_agent_executor.invoke({"input":str(prompt)})
    return response['output']
        

# st.write("Search Agent")
st.title("Search Agent")

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
    with st.spinner('Generating..'):
        chat_history = "\n".join([f"{history['role']}:{history['content']}" for history in st.session_state.messages])
        chat_history += prompt
        # print("\n".join([f"{data['role']}:{data['content']}" for data in st.session_state.messages]))
        response = ask(chat_history)
    # st.success("Done!")
    # st.success('Done!')
    # Assistant chat write
    with st.chat_message('assistant'):
        st.markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})