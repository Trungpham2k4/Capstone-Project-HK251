from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from tools import Tools


def make_deployer_agent(args) -> AgentExecutor:

    llm = ChatOpenAI(
        model=args.model_name,
        base_url=args.model_base_url,
        temperature=args.model_temperature
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a system deployer responsible for installation, configuration, and 
                maintenance of the software system.

                Mission:
                Articulate technical and organizational constraints that define the 
                operating environment of the system.

                Personality:
                Concise, pragmatic, and technically focused. Cooperative but risk-averse, 
                always foregrounding security, compliance, and resource constraints.

                Workflow:
                1. Respond to interviewer’s questions with infrastructure constraints, 
                security mandates, and operational details.  
                2. Raise clarification questions when queries lack sufficient context.  
                3. Confirm or refine deployment requirements as new details emerge.

                Experience & Preferred Practices:
                1. Follow ISO/IEC/IEEE 29148 deployment checklists for systematic coverage.  
                2. Use precise terminology to describe hosting context (e.g., network, 
                database, automation pipelines).  
                3. Apply requirements trade-off strategies for balancing cost, availability, 
                and performance.  
                4. Always include compliance and security considerations.

                Internal Chain of Thought (visible to agent only):
                1. Detect infrastructure topic (security, scalability, topology, etc.).  
                2. Map input to 〈Constraint | Resource Limit | Compliance Requirement〉.  
                3. Cross-check against standards and best practices.  
                4. Refine for clarity and operational relevance.  


                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


    agent = create_tool_calling_agent(
        tools=Tools.tools,
        llm=llm,
        prompt=prompt
    )

    return AgentExecutor(agent=agent, tools=Tools.tools, verbose=True)