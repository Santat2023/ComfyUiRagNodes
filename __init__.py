# импорт всех необходимых классов узлов из соответствующих 
# файлов чтобы они были доступны для использования в ComfyUI
from .nodes.llm_node import LLM_Node
from .nodes.db_load_node import DB_Load_Node

NODE_CLASS_MAPPINGS = {
    "MyNodesForLLM": LLM_Node,
    "MyNodesForDB": DB_Load_Node
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNodesForLLM": "LLM Node",
    "MyNodesForDB": "DB Load Node"
}
