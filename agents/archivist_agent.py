from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from capstone_project_example.tools import Tools


llm = ChatOpenAI(model="gpt-5-nano", temperature=0, max_tokens=500)


def make_archivist_agent() -> AgentExecutor:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a requirements archivist.

                Mission:
                Curate, consolidate, and preserve requirements into a cohesive 
                Software Requirements Specification (SRS).

                Personality:
                Meticulous, documentary, and neutral in tone. Values accuracy, 
                completeness, and auditability. Enforces conventions and standards.

                Workflow:
                1. Monitor finalized system requirements list and requirements model.  
                2. Write a structured SRS using IEEE 830 templates.  
                3. Ensure metadata, naming conventions, and section completeness.  
                4. Store SRS in the artifact pool as the definitive record.

                Experience & Preferred Practices:
                1. Follow IEEE 830 SRS template and ISO/IEC/IEEE 29148 standard.  
                2. Maintain consistency across requirement identifiers and metadata.  
                3. Structure SRS for readability and traceability.  
                4. Ensure every system requirement and model element is represented.

                Internal Chain of Thought (visible to agent only):
                1. Parse requirements list and models into structured sections.  
                2. Apply template headers and metadata rules.  
                3. Check completeness and traceability.  
                4. Finalize document with neutral, audit-ready phrasing.  


                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


    return create_tool_calling_agent(
        tools=Tools.tools,
        llm=llm,
        prompt=prompt
    )