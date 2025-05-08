
from typing import Optional

class RAGProcessor:
    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        
    def update_decision_tree_prompt(self, query: str, decision_tree: DecisionTree) -> str:
        """Generate prompt for decision tree update"""
        return f"""
        Update the {decision_tree.field} decision tree based on user query:
        Query: {query}
        Current Decision Tree: {decision_tree.nodes}
        
        Requirements:
        1. Maintain original structure integrity
        2. Only add/modify nodes relevant to the query
        3. Output updated tree in JSON format
        4. Highlight changes with 'MODIFIED' flag
        """
    
    def extract_subdescriptions_prompt(self, query: str, pkg: PseudoProgramKnowledgeGraph) -> str:
        """Generate prompt for subdescription extraction"""
        return f"""
        Extract relevant subdescriptions from knowledge graph:
        Query: {query}
        Knowledge Graph Entities: {pkg.entities}
        
        Requirements:
        1. Identify maximum all key entities
        2. Extract relationships between identified entities
        3. Output as structured subgraph in JSON
        4. Include confidence scores for each extraction
        """
    
    def process_query(self, query: str, 
                     decision_tree: DecisionTree, 
                     pkg: PseudoProgramKnowledgeGraph) -> tuple:
        """Full RAG processing pipeline"""
        # Update decision tree
        update_prompt = self.update_decision_tree_prompt(query, decision_tree)
        updated_tree = self.llm.query(update_prompt)
        
        # Extract subdescriptions
        extraction_prompt = self.extract_subdescriptions_prompt(query, pkg)
        updated_pkg = self.llm.query(extraction_prompt)
        
        return updated_tree, updated_pkg