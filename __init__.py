# импорт всех необходимых классов узлов из соответствующих 
# файлов чтобы они были доступны для использования в ComfyUI
from .custom_ksampler import RAG_KSampler_Node
from .llm_node import LLM_Node
from .db_load_node import DB_Load_Node

NODE_CLASS_MAPPINGS = {
    "MyNodesForRAG": RAG_KSampler_Node,
    "MyNodesForLLM": LLM_Node,
    "MyNodesForDB": DB_Load_Node
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MyNodesForRAG": "RAG KSampler Node",
    "MyNodesForLLM": "LLM Node",
    "MyNodesForDB": "DB Load Node"
}
