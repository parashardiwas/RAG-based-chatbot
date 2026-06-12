import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.services.rag.embedder import EmbeddingService
from app.services.rag.retriever import RetrievalService

async def main():
    emb = EmbeddingService()
    ret = RetrievalService()
    
    q1 = "what are the derivations of equations of motion ?"
    q2 = "what are derivations of equations of motion ?"
    
    v1 = await emb.embed_query(q2)
    
    res = await ret.retrieve_qa_pairs(
        query_embedding=v1,
        query_text=q2,
        top_k=5
    )
    
    print("Results for q2:")
    for r in res:
        print(f"  Score: {r.similarity_score:.4f} | Content: {r.content}")

asyncio.run(main())
