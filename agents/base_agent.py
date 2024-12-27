from abc import ABC

class BaseAgent(ABC):

    def draw_png(self):
        pass

    def run(self, query: str):
        pass

    def stream(self, query: str):
        pass