from knowledge_graph import KnowledgeGraphGenerator
from rag_processor import RAGProcessor
from execution_engine import ExecutionEngine
import LLMAdapter

def initialize_expert_decision_tree(field: str) -> DecisionTree:
    """Expert-built initial decision tree (simulated)"""
    return DecisionTree(
        field=field,
        nodes={
            "root": {"children": ["node1", "node2"], "description": "Initial decision root"},
            # ... other nodes
        }
    )

if __name__ == "__main__":
    # Initialize components
    llm = LLMAdapter(api_key="your_llm_api_key")
    field = "medical_diagnosis"
    
    # Stage 1: Knowledge Graph Generation
    kg_gen = KnowledgeGraphGenerator(llm)
    decision_tree = initialize_expert_decision_tree(field)
    initial_pkg = kg_gen.generate_initial_pkg(decision_tree, expert_modify=True)
    
    # Stage 2: RAG Processing
    rag = RAGProcessor(llm)
    user_query = "Patient with fever and chest pain, history of diabetes"
    updated_tree, updated_pkg = rag.process_query(user_query, decision_tree, initial_pkg)
    
    # Stage 3: Execution
    executor = ExecutionEngine(llm)
    final_result = executor.execute(updated_pkg)
    
    print("Final Decision:\n", final_result)