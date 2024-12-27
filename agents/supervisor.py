import operator
from pydantic import BaseModel, Field
from typing import Annotated, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from .agent_role import AgentRoleManager, AgentRole
from .base_agent import BaseAgent

ROLES: list[AgentRole] = [
    AgentRole(index=1, name="一般エキスパート", description="幅広い分野の一般的な質問に答える", details="幅広い分野の一般的な質問に対して、正確でわかりやすい回答を提供してください"),
    AgentRole(index=2, name="生成AI製品エキスパート", description="生成AIや関連製品、技術に関する専門的な質問に答える", details="生成AIや関連製品、技術に関する専門的な質問に対して、最新の情報と深い洞察を提供してください。"),
    AgentRole(index=3, name="カウンセラー", description="個人的な悩みや心理的な問題に対してサポートを提供する", details="個人的な悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能であれば適切なアドバイスも行ってください"),
]

class Supervisor(BaseAgent):
    """
    特徴：
        - 質問にいちばん適したエージェントを選定し、そのエージェントが回答を生成する。品質に問題があれば再度回答を生成する

    エージェント一覧：
        - 一般エキスパート
        - 生成AI製品エキスパート
        - カウンセラー
    """
    GRAPH_PNG_PATH = "./out/multi-ai-agent.png"

    def __init__(self, llm: Any):

        # n = Node(llm) # 同期
        n = Node(llm) # 非同期
        workflow = StateGraph(State)
        workflow.add_node("selection", n.selection_node)
        workflow.add_node("answering", n.answering_node)
        workflow.add_node("check", n.check_node)

        workflow.set_entry_point("selection")
        workflow.add_edge("selection", "answering")
        workflow.add_edge("answering", "check")
        workflow.add_conditional_edges(
            "check",
            lambda state: state.current_judge,
            {True: END, False: "selection"}
        )

        self.compiled = workflow.compile()

    def draw_png(self):
        self.compiled.get_graph().draw_png(output_file_path=self.GRAPH_PNG_PATH)

    def run(self, query: str):
        initial_state = State(query=query)
        result = self.compiled.invoke(initial_state)
        return result

    def stream(self, query: str):
        initial_state = State(query=query)
        return self.compiled.stream(initial_state)

class State(BaseModel):
    query: str = Field(..., description="ユーザからの質問")
    current_roles: str = Field(default="", description="選定された回答ロール")
    messages: Annotated[list[str], operator.add] = Field(default=[], description="回答履歴")
    current_judge: bool = Field(default=False, description="品質チェックの結果")
    judgement_reason: str = Field(default="", description="品質チェックの判定理由")

class Judgement(BaseModel):
    reason: str = Field(default="", description="判定理由")
    judge: bool = Field(default=False, description="判定結果")

class Node:
    def __init__(self, llm):
        self.llm = llm

    def selection_node(self, state: State) -> dict[str, Any]:
        query = state.query
        role_options = "\n".join(f"{r.index}. {r.description}" for r in ROLES)
        prompt = ChatPromptTemplate.from_template(
f"""質問を分析し、最も適切な回答担当ロールを選択してください。

{role_options}

回答は選択肢の番号(1,2, または3)のみを返してください。

質問： {query}
""".strip()
        )
        chain = prompt | self.llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
        role_number = chain.invoke({"role_options": role_options, "query": query})
        selected_role: AgentRole = AgentRoleManager.get_role_from_number(ROLES, role_number)
        return {"current_role": selected_role.name}

    def answering_node(self, state: State) -> dict[str, Any]:
        query= state.query
        role = state.current_roles
        role_details = "\n".join(f"- {r.name}: {r.details}" for r in ROLES)
        prompt = ChatPromptTemplate.from_template(
f"""あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。

役割の詳細：
{role_details}

質問： {query}
回答：""".strip()
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"role": role, "role_details": role_details, "query": query})
        return {"messages": [answer]}

    def check_node(self, state: State) -> dict[str, Any]:
        query = state.query
        answer = state.messages[-1]
        prompt = ChatPromptTemplate.from_template(
f"""以下の回答の品質をチェックし、問題がある場合は'False', 問題がない場合は'True'を回答してください。また、その判断理由も説明してください。


ユーザからの質問： {query}
回答：{answer}
""".strip()
        )
        chain = prompt | self.llm.with_structured_output(Judgement)
        result: Judgement = chain.invoke({"query": query, "answer": answer})

        return {"current_judge": result.judge, "judgement_reason": result.reason}
