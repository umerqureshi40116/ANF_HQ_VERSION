import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv
import os
import re
from typing import List, Tuple, Optional, Dict

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables")
    st.stop()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")

class LegalChatbot:
    def __init__(self, chroma_path: str):
        self.db = self._load_database(chroma_path)
        self.llm = self._load_llm()
        self.law_keywords = self._init_law_keywords()
        self.query_enhancements = self._init_query_enhancements()
    
    @staticmethod
    @st.cache_resource
    def _load_database(chroma_path: str) -> Chroma:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
            collection_name="legal_sections"
        )
    
    @staticmethod
    @st.cache_resource
    def _load_llm() -> ChatGroq:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
    
    @staticmethod
    def _init_law_keywords() -> Dict[str, List[str]]:
        return {
            "CNS": [
                "narcotic", "drug", "opium", "heroin", "cannabis",
                "controlled substance", "psychotropic", "trafficking",
                "cultivation", "possession", "cocaine", "hashish"
            ],
            "PPC": [
                "penal code", "murder", "theft", "assault",
                "culpable homicide", "hurt", "kidnapping",
                "criminal conspiracy", "defamation"
            ],
            "POLICE": [
                "police rules", "officer", "duty", "patrol",
                "jail", "prisoner", "gazetted"
            ],
            "AMLA": [
                "money laundering", "anti-money laundering",
                "proceeds of crime", "suspicious transaction"
            ],
            "ANF": [
                "anti narcotics force", "anf officer", "narcotics operations"
            ],
            "QES": [
                "evidence", "testimony", "witness", "proof",
                "qanun-e-shahadat"
            ]
        }
    
    @staticmethod
    def _init_query_enhancements() -> Dict[str, str]:
        return {
            "heroin": "heroin diacetylmorphine opium derivative possession trafficking",
            "trafficking": "trafficking transport export import section 9 penalty",
            "possession": "possession narcotic substance section 6 imprisonment",
            "cultivation": "cultivation cannabis opium poppy section 4 prohibition",
            "manufacture": "manufacture production substance section 10 equipment",
            "death penalty": "death penalty life imprisonment trafficking kilograms",
            "bail": "bail section 51 special court no bail death penalty",
            "arrest": "arrest warrant search seizure section 20 authority",
            "treatment": "treatment rehabilitation addict centers registration",
            "money laundering": "money laundering proceeds crime financial monitoring"
        }
    
    def detect_law_context(self, query: str) -> str:
        query_lower = query.lower()
        counts = {}
        
        for law, keywords in self.law_keywords.items():
            counts[law] = sum(1 for kw in keywords if kw in query_lower)
        
        max_count = max(counts.values())
        if max_count == 0:
            return "GENERAL"
        
        return max(counts, key=counts.get)
    
    def extract_section_number(self, query: str) -> Optional[str]:
        patterns = [
            r"\bsection\s+(\d+\.\d+[A-Z]?)\b",
            r"\bsec\.?\s+(\d+\.\d+[A-Z]?)\b",
            r"\brule\s+(\d+\.\d+[A-Z]?)\b",
            r"\bsection\s+(\d+[A-Z]?)\b",
            r"\brule\s+(\d+[A-Z]?)\b",
            r"\barticle\s+(\d+[A-Z]?)\b",
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    
    def search_by_section_number(
        self, section_num: str, law_filter: Optional[str] = None
    ) -> Optional[List[Tuple[Document, float]]]:
        try:
            results = self.db.get(where={"section": str(section_num)})
            if not results or not results.get("documents"):
                return None
            
            docs_with_scores = []
            for i, doc_text in enumerate(results["documents"]):
                metadata = results.get("metadatas", [{}])[i] if results.get("metadatas") else {}
                
                if law_filter and law_filter != "GENERAL":
                    law_name = metadata.get("law_name", "").lower()
                    if not self._matches_law_filter(law_name, law_filter):
                        continue
                
                doc = Document(page_content=doc_text, metadata=metadata)
                docs_with_scores.append((doc, 0.0))
            
            return docs_with_scores if docs_with_scores else None
        except Exception as e:
            st.warning(f"Metadata search error: {e}")
            return None
    
    def _matches_law_filter(self, law_name: str, law_filter: str) -> bool:
        filters = {
            "CNS": "narcotic",
            "PPC": "penal code",
            "POLICE": "police rules",
            "AMLA": "money laundering",
            "ANF": "anf act",
            "QES": "shahadat"
        }
        return filters.get(law_filter, "").lower() in law_name
    
    def enhance_query(self, query: str) -> str:
        query_lower = query.lower()
        enhancements = []
        
        for term, enhancement in self.query_enhancements.items():
            if term in query_lower:
                enhancements.append(enhancement)
        
        if enhancements:
            return f"{query} {' '.join(enhancements)}"
        return query
    
    def semantic_search(
        self,
        query: str,
        k: int = 25,
        score_threshold: float = 2.0
    ) -> List[Tuple[Document, float]]:
        enhanced_query = self.enhance_query(query)
        results = self.db.similarity_search_with_score(enhanced_query, k=k)
        return [(doc, score) for doc, score in results if score < score_threshold]
    
    def multi_strategy_search(
        self,
        query: str,
        law_context: str,
        section_num: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        all_results = []
        seen_keys = set()
        
        if section_num:
            direct_results = self.search_by_section_number(section_num, law_context)
            if direct_results:
                for doc, score in direct_results:
                    key = (doc.metadata.get("law_name"), doc.metadata.get("section"))
                    if key not in seen_keys:
                        all_results.append((doc, score))
                        seen_keys.add(key)
        
        semantic_results = self.semantic_search(query, k=30, score_threshold=2.0)
        for doc, score in semantic_results:
            key = (doc.metadata.get("law_name"), doc.metadata.get("section"))
            if key not in seen_keys:
                all_results.append((doc, score))
                seen_keys.add(key)
        
        if law_context != "GENERAL":
            all_results = [
                (doc, score) for doc, score in all_results
                if self._matches_law_filter(
                    doc.metadata.get("law_name", "").lower(),
                    law_context
                )
            ]
        
        all_results = sorted(all_results, key=lambda x: x[1])
        return all_results[:20]


def format_prompt(context: str, history: str, question: str) -> str:
    return f"""You are a legal assistant specializing in Pakistani law:
- Pakistan Penal Code (PPC), 1860
- Control of Narcotic Substances Act (CNSA), 1997
- Punjab Police Rules, 1934
- Anti-Money Laundering Act (AMLA), 2010
- Anti Narcotics Force Act, 1997
- Qanun-e-Shahadat Order, 1984

INSTRUCTIONS:
1. Answer ONLY using provided context
2. Always cite specific sections with full law names
3. If information is unavailable, state clearly
4. Use professional, clear language
5. Include relevant penalties, definitions, procedures

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

QUESTION:
{question}

ANSWER:"""


def main():
    st.set_page_config(page_title="Legal Chatbot", layout="wide")
    st.title("Legal Document Chatbot")
    st.caption("RAG-powered Q&A for Pakistani legal documents")
    
    chatbot = LegalChatbot(CHROMA_PATH)
    
    with st.sidebar:
        st.header("About")
        st.info("""
        Ask questions about:
        - Pakistan Penal Code (PPC)
        - Control of Narcotic Substances Act (CNSA)
        - Punjab Police Rules, 1934
        - Anti-Money Laundering Act (AMLA)
        - Anti Narcotics Force Act
        - Qanun-e-Shahadat Order
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.text(f"{src['law']} - Section {src['section']}: {src['title']}")
    
    if query := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        history = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages[-10:-1]
        ])
        
        with st.spinner("Searching..."):
            law_context = chatbot.detect_law_context(query)
            section_num = chatbot.extract_section_number(query)
            results = chatbot.multi_strategy_search(query, law_context, section_num)
        
        context_parts = []
        sources = []
        seen = set()
        
        for doc, score in results[:12]:
            sec = doc.metadata.get("section", "Unknown")
            law = doc.metadata.get("law_name", "Unknown")
            key = f"{law}:{sec}"
            
            if key in seen:
                continue
            seen.add(key)
            
            title = doc.metadata.get("title", "")
            context_parts.append(f"{law}\nSection {sec}: {title}\n{doc.page_content}\n")
            
            sources.append({
                "section": sec,
                "title": title,
                "law": law,
                "score": round(score, 3)
            })
            
            if len(sources) >= 8:
                break
        
        context_text = "\n".join(context_parts) if context_parts else "No results found."
        prompt = format_prompt(context_text, history, query)
        
        with st.spinner("Generating response..."):
            try:
                response = chatbot.llm.invoke(prompt).content
            except Exception as e:
                st.error(f"Error: {e}")
                response = "Unable to generate response. Please try again."
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
        
        with st.chat_message("assistant"):
            st.markdown(response)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.text(f"{src['law']} - Section {src['section']}: {src['title']}")


if __name__ == "__main__":
    main()
