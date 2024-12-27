from pydantic import BaseModel

class RoleNotFoundError(Exception):
    pass

class AgentRole(BaseModel):
        index: int
        name: str
        description: str
        details: str

class AgentRoleManager:
    @classmethod
    def get_role_from_number(cls, roles: list[AgentRole], target_number: str) -> AgentRole:
        for r in roles:
            if r.index == int(target_number):
                return r

        raise RoleNotFoundError(f"Role not found index:'{target_number}'.")