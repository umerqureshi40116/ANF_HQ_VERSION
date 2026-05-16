from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import re

load_dotenv()

app = FastAPI(title="Legal RAG Chatbot API", version="1.0.0")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable not set")


class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[dict]] = None
    law_context: Optional[str] = None


class SearchResult(BaseModel):
    section: str
    title: str
    law: str
    content: str
    relevance_score: float


class ChatResponse(BaseModel):
    response: str
    sources: List[SearchResult]


class LegalRAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name="legal_sections"
        )
        
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
    
    def detect_law_context(self, query: str) -> str:
        law_keywords = {
            "CNS": ["narcotic", "drug", "opium", "heroin", "cannabis", "trafficking", "possession"],
            "PPC": ["penal code", "murder", "theft", "assault", "culpable homicide"],
            "POLICE": ["police rules", "officer", "duty", "jail", "prisoner"],
            "AMLA": ["money laundering", "proceeds", "suspicious transaction"],
            "ANF": ["anti narcotics force", "anf officer"],
            "QES": ["evidence", "testimony", "witness"]
        }
        
        query_lower = query.lower()
        counts = {law: sum(1 for kw in keywords if kw in query_lower) 
                 for law, keywords in law_keywords.items()}
        
        max_count = max(counts.values())
        return max(counts, key=counts.get) if max_count > 0 else "GENERAL"
    
    def extract_section_number(self, query: str) -> Optional[str]:
        patterns = [
            r"\bsection\s+(\d+\.\d+[A-Z]?)\b",
            r"\bsec\.?\s+(\d+\.\d+[A-Z]?)\b",
            r"\brule\s+(\d+\.\d+[A-Z]?)\b",
            r"\bsection\s+(\d+[A-Z]?)\b",
            r"\brule\s+(\d+[A-Z]?)\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower(), re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    
    def semantic_search(
        self,
        query: str,
        k: int = 25,
        score_threshold: float = 2.0
    ) -> List[tuple]:
        results = self.db.similarity_search_with_score(query, k=k)
        return [(doc, score) for doc, score in results if score < score_threshold]
    
    def rerank_results(
        self,
        results: List[tuple],
        query: str,
        law_context: str = "GENERAL"
    ) -> List[tuple]:
        """Rerank results by relevance score and law context."""
        law_boosts = {
            "CNS": lambda law: 0.8 if "narcotic" in law.lower() else 1.0,
            "PPC": lambda law: 0.8 if "penal code" in law.lower() else 1.0,
            "POLICE": lambda law: 0.8 if "police rules" in law.lower() else 1.0,
            "AMLA": lambda law: 0.8 if "money laundering" in law.lower() else 1.0,
            "ANF": lambda law: 0.8 if "anf act" in law.lower() else 1.0,
            "QES": lambda law: 0.8 if "shahadat" in law.lower() else 1.0,
            "GENERAL": lambda law: 1.0
        }
        
        boost_fn = law_boosts.get(law_context, lambda law: 1.0)
        
        reranked = [
            (doc, score * boost_fn(doc.metadata.get("law_name", "")))
            for doc, score in results
        ]
        
        return sorted(reranked, key=lambda x: x[1])
    
    def retrieve_context(
        self,
        query: str,
        law_context: str = "GENERAL",
        section_num: Optional[str] = None,
        max_sources: int = 8
    ) -> tuple:
        """Retrieve and rerank context for query."""
        
        all_results = []
        seen_keys = set()
        
        if section_num:
            try:
                direct_results = self.db.get(where={"section": str(section_num)})
                if direct_results and direct_results.get("documents"):
                    for i, doc_text in enumerate(direct_results["documents"]):
                        metadata = direct_results.get("metadatas", [{}])[i]
                        doc = Document(page_content=doc_text, metadata=metadata)
                        key = (metadata.get("law_name"), section_num)
                        if key not in seen_keys:
                            all_results.append((doc, 0.0))
                            seen_keys.add(key)
            except Exception:
                pass
        
        semantic_results = self.semantic_search(query, k=30)
        for doc, score in semantic_results:
            key = (doc.metadata.get("law_name"), doc.metadata.get("section"))
            if key not in seen_keys:
                all_results.append((doc, score))
                seen_keys.add(key)
        
        reranked = self.rerank_results(all_results, query, law_context)
        
        sources = []
        context_parts = []
        
        for doc, score in reranked[:max_sources]:
            section = doc.metadata.get("section", "Unknown")
            law = doc.metadata.get("law_name", "Unknown")
            title = doc.metadata.get("title", "")
            
            context_parts.append(f"{law}\nSection {section}: {title}\n{doc.page_content}")
            sources.append(SearchResult(
                section=section,
                title=title,
                law=law,
                content=doc.page_content,
                relevance_score=round(score, 3)
            ))
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        return context, sources
    
    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in conversation_history[-10:]
            ])
        
        prompt = f"""You are a legal assistant specializing in Pakistani law.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

QUESTION:
{query}

ANSWER:"""
        
        return self.llm.invoke(prompt).content


rag_engine = LegalRAGEngine()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        law_context = request.law_context or rag_engine.detect_law_context(request.query)
        section_num = rag_engine.extract_section_number(request.query)
        
        context, sources = rag_engine.retrieve_context(
            request.query,
            law_context,
            section_num,
            max_sources=8
        )
        
        response_text = rag_engine.generate_response(
            request.query,
            context,
            request.conversation_history
        )
        
        return ChatResponse(response=response_text, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(query: str, k: int = 10, law_context: str = "GENERAL"):
    try:
        section_num = rag_engine.extract_section_number(query)
        context, sources = rag_engine.retrieve_context(query, law_context, section_num, k)
        return {"sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
