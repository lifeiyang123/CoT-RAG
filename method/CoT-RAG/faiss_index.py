import numpy as np
import faiss
from typing import List, Dict, Tuple

class FaissRAG:
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize FaissRAG class
        :param embedding_dim: dimensionality of the embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.index_type = None
        self.documents = []
    
    def create_index(self, index_type: str, **kwargs):
        """
        Create specified Faiss index type
        :param index_type: index type, supports 7 variants:
            - "FlatL2": exact L2 distance search
            - "FlatIP": exact inner product search
            - "HNSW": graph-based approximate search
            - "IVFFlat": inverted file with exact search
            - "LSH": locality-sensitive hashing
            - "PQ": product quantization
            - "IVFPQ": inverted file with product quantization
        :param kwargs: specific parameters for different index types
        """
        self.index_type = index_type
        
        if index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        elif index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        elif index_type == "HNSW":
            # HNSW parameters
            M = kwargs.get('M', 32)  # number of connections per node
            efConstruction = kwargs.get('efConstruction', 200)  # construction-time search scope
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.index.hnsw.efConstruction = efConstruction
        
        elif index_type == "IVFFlat":
            nlist = kwargs.get('nlist', 100)  # number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        
        elif index_type == "LSH":
            nbits = kwargs.get('nbits', 8)  # number of hash bits
            self.index = faiss.IndexLSH(self.embedding_dim, nbits)
        
        elif index_type == "PQ":
            M = kwargs.get('M', 8)  # number of subspaces
            nbits = kwargs.get('nbits', 8)  # bits per subspace
            self.index = faiss.IndexPQ(self.embedding_dim, M, nbits)
        
        elif index_type == "IVFPQ":
            nlist = kwargs.get('nlist', 100)  # number of clusters
            M = kwargs.get('M', 8)  # number of subspaces
            nbits = kwargs.get('nbits', 8)  # bits per subspace
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, M, nbits)
        
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        """
        Add documents and corresponding embeddings to the index
        :param embeddings: document embedding matrix (n, embedding_dim)
        :param documents: list of document contents
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        # For index types that require training
        if self.index_type in ["IVFFlat", "PQ", "IVFPQ"] and not self.index.is_trained:
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for top-k most relevant documents
        :param query_embedding: query embedding vector (1, embedding_dim)
        :param k: number of documents to return
        :return: list of document contents and similarity scores
        """
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            # For IVF indexes, set nprobe for search
            nprobe = min(10, self.index.nlist)  # number of clusters to search
            self.index.nprobe = nprobe
        
        if self.index_type == "HNSW":
            # For HNSW index, set efSearch
            self.index.hnsw.efSearch = 100
        
        # Ensure query vector is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        # For inner product similarity, convert distances to scores
        if self.index_type == "FlatIP":
            scores = distances[0]
        else:
            scores = 1 / (1 + distances[0])  # convert distance to similarity
        
        # Return documents and scores
        return [(self.documents[i], scores[idx]) for idx, i in enumerate(indices[0])]


# Example usage
if __name__ == "__main__":
    # Sample documents and embeddings
    documents = []
    
    # Generate random embeddings for demonstration (use real embeddings in practice)
    np.random.seed(42)
    embeddings = np.random.rand(len(documents), 768).astype('float32')
    query_embedding = np.random.rand(1, 768).astype('float32')
    
    # Test all index types
    index_types = ["FlatL2", "FlatIP", "HNSW", "IVFFlat", "LSH", "PQ", "IVFPQ"]
    
    for index_type in index_types:
        print(f"\n=== Testing {index_type} index ===")
        
        # Create RAG instance
        rag = FaissRAG(embedding_dim=768)
        
        # Create specific index type
        if index_type == "IVFFlat":
            rag.create_index(index_type, nlist=5)
        elif index_type == "IVFPQ":
            rag.create_index(index_type, nlist=5, M=8, nbits=8)
        elif index_type == "HNSW":
            rag.create_index(index_type, M=16, efConstruction=200)
        else:
            rag.create_index(index_type)
        
        # Add documents
        rag.add_documents(embeddings, documents)
        
        # Perform search
        results = rag.search(query_embedding, k=2)
        
        # Print results
        for doc, score in results:
            print(f"Score: {score:.4f} - Document: {doc[:50]}...")