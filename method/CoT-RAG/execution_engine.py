
class ExecutionEngine:
    def __init__(self, llm_adapter):
        self.llm = llm_adapter
        
    def generate_execution_prompt(self, pkg: PseudoProgramKnowledgeGraph) -> str:
        """Generate final execution prompt"""
        return f"""
        Generate final output using pseudo-program knowledge graph:
        {pkg.entities}
        {pkg.relationships}
        
        Requirements:
        1. Structure output in markdown format
        2. Include decision path visualization
        3. Provide confidence assessments
        4. List supporting evidence from knowledge graph
        5. Highlight any expert modifications
        """
    
    def execute(self, pkg: PseudoProgramKnowledgeGraph) -> str:
        """Execute the final reasoning"""
        prompt = self.generate_execution_prompt(pkg)
        return self.llm.query(prompt)