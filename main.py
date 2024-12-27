from argparse import ArgumentParser
from agents import Supervisor, AgentType
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1")

def arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--graph", "-g", action="store_true", help="Generate AI-Agent graph.")
    parser.add_argument("--type", "-t", type=AgentType, help="Select the AI-Agent type.", choices=AgentType, required=True)
    args = parser.parse_args()
    return args

def resolve_agent_type(agent_type: str):
    match agent_type:
        case AgentType.SUPERVISOR:
            return Supervisor(llm)
        case _:
            raise ValueError(f"Agent Type was not matched. TYPE:{agent_type}")

def main():
    args = arg_parser()

    query = "生成AIについて教えてください"

    # Agentの種類を特定
    agent = resolve_agent_type(args.type)

    if args.graph:
        agent.draw_png()
        exit()

    result = agent.run(query)
    print(result)



if __name__ == "__main__":
    main()