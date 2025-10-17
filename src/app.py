# # # import streamlit as st
# # # from langchain_chroma import Chroma
# # # from langchain_groq import ChatGroq
# # # from langchain_huggingface import HuggingFaceEmbeddings
# # # from langchain.prompts import ChatPromptTemplate
# # # from dotenv import load_dotenv
# # # import os
# # # import re

# # # # ---------------- Load environment variables ----------------
# # # load_dotenv()
# # # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # # if not GROQ_API_KEY:
# # #     st.error("⚠️ GROQ_API_KEY not found! Please set it in your .env file.")
# # #     st.stop()

# # # # ---------------- Paths ----------------
# # # CHROMA_PATH = r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\chroma"

# # # # ---------------- Initialize Chroma with matching embeddings ----------------
# # # @st.cache_resource
# # # def load_database():
# # #     hf_embeddings = HuggingFaceEmbeddings(
# # #         model_name="sentence-transformers/all-mpnet-base-v2",
# # #         model_kwargs={'device': 'cpu'},
# # #         encode_kwargs={'normalize_embeddings': True}
# # #     )
# # #     return Chroma(
# # #         persist_directory=CHROMA_PATH,
# # #         embedding_function=hf_embeddings,
# # #         collection_name="legal_sections"
# # #     )

# # # db = load_database()

# # # # ---------------- Law Detection ----------------
# # # def detect_law_context(query):
# # #     """Detect which law the user is asking about."""
# # #     query_lower = query.lower()
    
# # #     # CNS/CNSA keywords
# # #     cns_keywords = ['cns', 'cnsa', 'narcotic', 'drug', 'opium', 'heroin', 
# # #                     'cannabis', 'controlled substance', 'psychotropic']
    
# # #     # PPC keywords
# # #     ppc_keywords = ['ppc', 'penal code', 'murder', 'theft', 'assault', 
# # #                     'culpable homicide', 'hurt', 'wrongful restraint']
    
# # #     has_cns = any(keyword in query_lower for keyword in cns_keywords)
# # #     has_ppc = any(keyword in query_lower for keyword in ppc_keywords)
    
# # #     if has_cns and not has_ppc:
# # #         return "CNS"
# # #     elif has_ppc and not has_cns:
# # #         return "PPC"
# # #     else:
# # #         return "BOTH"  # Return both if ambiguous or both mentioned

# # # # ---------------- Section Number Extraction ----------------
# # # def extract_section_number(query):
# # #     """Extract section number from queries."""
# # #     patterns = [
# # #         r'\bsection\s+(\d+[A-Z]?)\b',
# # #         r'\bsec\.?\s+(\d+[A-Z]?)\b',
# # #         r'\bs\.?\s+(\d+[A-Z]?)\b',
# # #         r'(?:^|\s)(\d+[A-Z]?)(?:\s|$|[.,!?])',
# # #     ]
    
# # #     query_lower = query.lower()
# # #     for pattern in patterns:
# # #         match = re.search(pattern, query_lower)
# # #         if match:
# # #             return match.group(1).upper()
# # #     return None

# # # def search_by_section_number(section_num, law_filter=None):
# # #     """
# # #     Direct metadata search for a specific section number.
# # #     If law_filter is specified, only return results from that law.
# # #     """
# # #     try:
# # #         # Get all documents with this section number
# # #         results = db.get(where={"section": str(section_num)})
        
# # #         if results and results['documents']:
# # #             docs_with_scores = []
# # #             for i, doc_text in enumerate(results['documents']):
# # #                 metadata = results['metadatas'][i] if results['metadatas'] else {}
                
# # #                 # Apply law filter if specified
# # #                 if law_filter and law_filter != "BOTH":
# # #                     law_name = metadata.get('law_name', '')
# # #                     if law_filter == "CNS" and "Narcotic" not in law_name:
# # #                         continue
# # #                     elif law_filter == "PPC" and "Penal Code" not in law_name:
# # #                         continue
                
# # #                 from langchain.schema import Document
# # #                 doc = Document(
# # #                     page_content=doc_text,
# # #                     metadata=metadata
# # #                 )
# # #                 docs_with_scores.append((doc, 0.0))
            
# # #             return docs_with_scores if docs_with_scores else None
# # #     except Exception as e:
# # #         st.warning(f"Metadata search failed: {e}")
    
# # #     return None

# # # # ---------------- Enhanced Prompt Template ----------------
# # # PROMPT_TEMPLATE = """You are a knowledgeable legal assistant specializing in Pakistani law, particularly:
# # # 1. Pakistan Penal Code (PPC), 1860
# # # 2. Control of Narcotic Substances Act (CNSA), 1997

# # # Your task is to provide accurate, helpful legal information based ONLY on the provided context.

# # # CRITICAL RULES:
# # # 1. Answer ONLY using information from the context below - NEVER use external knowledge
# # # 2. ALWAYS cite specific sections AND the law name (e.g., "According to Section 9 of the Control of Narcotic Substances Act...")
# # # 3. If a section number appears in multiple laws, mention ALL relevant sections
# # # 4. If the exact answer is not in the context, respond: "I apologize, but I don't have information about that specific provision in the available legal documents. Please try rephrasing your question or ask about a different section."
# # # 5. Use clear, professional language suitable for educational purposes
# # # 6. When referencing sections, include the section number, title, AND law name
# # # 7. Quote directly from the section text when appropriate

# # # CONTEXT (Relevant Legal Sections):
# # # {context}

# # # CONVERSATION HISTORY:
# # # {history}

# # # QUESTION: {question}

# # # ANSWER (with mandatory section and law citations):"""

# # # # ---------------- Initialize LLM ----------------
# # # @st.cache_resource
# # # def load_llm():
# # #     return ChatGroq(
# # #         api_key=GROQ_API_KEY,
# # #         model="llama-3.3-70b-versatile",
# # #         temperature=0.1
# # #     )

# # # model = load_llm()

# # # # ---------------- Streamlit UI ----------------
# # # st.set_page_config(page_title="ANF EduBot", page_icon="🧑‍⚖️", layout="wide")

# # # # Header
# # # st.title("🧑‍⚖️ ANF Academy Educational Chatbot")
# # # st.caption("Ask questions about Pakistani Law (PPC & CNSA)")

# # # # Sidebar with info
# # # with st.sidebar:
# # #     st.header("📚 About")
# # #     st.info("""
# # #     This chatbot provides information about:
# # #     - **Pakistan Penal Code (PPC), 1860**
# # #     - **Control of Narcotic Substances Act (CNSA), 1997**
    
# # #     **Tips for best results:**
# # #     - Specify the law: "What is Section 9 of CNSA?"
# # #     - Ask about offenses: "What are punishments for drug trafficking?"
# # #     - Ask about definitions: "What does narcotics mean in CNSA?"
# # #     - Be specific in your questions
# # #     """)
    
# # #     st.header("🔍 Database Stats")
# # #     try:
# # #         collection = db._collection
# # #         total = collection.count()
# # #         st.metric("Total Chunks", total)
        
# # #         # Show breakdown by law
# # #         all_docs = db.get()
# # #         from collections import Counter
# # #         law_counts = Counter(m.get('law_name') for m in all_docs['metadatas'])
        
# # #         st.write("**By Law:**")
# # #         for law, count in law_counts.items():
# # #             st.write(f"• {law}: {count} chunks")
# # #     except Exception as e:
# # #         st.write("Database loaded ✅")
    
# # #     if st.button("🗑️ Clear Chat History"):
# # #         st.session_state.messages = []
# # #         st.rerun()

# # # # Initialize chat history
# # # if "messages" not in st.session_state:
# # #     st.session_state.messages = []

# # # # Display chat history
# # # for msg in st.session_state.messages:
# # #     with st.chat_message(msg["role"]):
# # #         st.markdown(msg["content"])
# # #         if "sources" in msg and msg["sources"]:
# # #             with st.expander("📚 View Sources"):
# # #                 for src in msg["sources"]:
# # #                     st.markdown(f"**{src['law']}** - Section {src['section']}: {src['title']} (Page {src['page']})")

# # # # User input
# # # if query := st.chat_input("Ask a legal question (e.g., 'What is Section 9 of CNSA?')"):
# # #     # Add user message
# # #     st.session_state.messages.append({"role": "user", "content": query})
# # #     with st.chat_message("user"):
# # #         st.markdown(query)

# # #     # Build conversation history (last 10 messages)
# # #     history_text = ""
# # #     recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
# # #     for msg in recent_messages[:-1]:
# # #         role = "User" if msg["role"] == "user" else "Assistant"
# # #         history_text += f"{role}: {msg['content']}\n"

# # #     # HYBRID RETRIEVAL STRATEGY
# # #     with st.spinner("🔍 Searching legal documents..."):
# # #         results = []
        
# # #         # Step 1: Detect law context
# # #         law_context = detect_law_context(query)
        
# # #         # Step 2: Check if query mentions a specific section number
# # #         section_num = extract_section_number(query)
        
# # #         if section_num:
# # #             st.info(f"🎯 Detected Section {section_num} - Law context: {law_context}")
# # #             # Try direct metadata search with law filter
# # #             direct_results = search_by_section_number(section_num, law_context)
# # #             if direct_results:
# # #                 results = direct_results
# # #                 st.success(f"✅ Found {len(results)} chunks for Section {section_num}")
        
# # #         # Step 3: If no direct results, use semantic search
# # #         if not results:
# # #             # Enhanced query based on law context
# # #             enhanced_query = query
# # #             if law_context == "CNS":
# # #                 enhanced_query = f"{query} narcotics drugs controlled substances CNSA"
# # #             elif law_context == "PPC":
# # #                 enhanced_query = f"{query} penal code crime offense PPC"
            
# # #             if section_num:
# # #                 enhanced_query = f"{enhanced_query} section {section_num}"
            
# # #             results = db.similarity_search_with_score(enhanced_query, k=15)
            
# # #             # Filter by relevance
# # #             results = [(doc, score) for doc, score in results if score < 1.5]
            
# # #             # Filter by law context if specified
# # #             if law_context in ["CNS", "PPC"]:
# # #                 filtered_results = []
# # #                 for doc, score in results:
# # #                     law_name = doc.metadata.get('law_name', '')
# # #                     if law_context == "CNS" and "Narcotic" in law_name:
# # #                         filtered_results.append((doc, score))
# # #                     elif law_context == "PPC" and "Penal Code" in law_name:
# # #                         filtered_results.append((doc, score))
# # #                     elif law_context == "BOTH":
# # #                         filtered_results.append((doc, score))
                
# # #                 if filtered_results:
# # #                     results = filtered_results
            
# # #             # Sort results: prioritize exact section matches, then by score
# # #             if section_num:
# # #                 def sort_key(item):
# # #                     doc, score = item
# # #                     is_exact = doc.metadata.get("section") == section_num
# # #                     return (not is_exact, score)
# # #                 results = sorted(results, key=sort_key)
        
# # #         if len(results) == 0:
# # #             st.warning("⚠️ No relevant sections found. Try:\n- Specifying the law (PPC or CNSA)\n- Specifying a section number\n- Rephrasing your question\n- Using keywords from the legal text")
    
# # #     # Build context with section references (limit to top 6)
# # #     context_parts = []
# # #     sources = []
# # #     seen_sections = set()
    
# # #     for doc, score in results[:10]:  # Get up to 10 to ensure diversity
# # #         section = doc.metadata.get("section", "Unknown")
# # #         law_name = doc.metadata.get("law_name", "Unknown Law")
        
# # #         # Create unique key combining law and section
# # #         section_key = f"{law_name}:{section}"
        
# # #         # Skip duplicate section+law combinations
# # #         if section_key in seen_sections:
# # #             continue
# # #         seen_sections.add(section_key)
        
# # #         title = doc.metadata.get("title", "")
# # #         page = doc.metadata.get("page", "N/A")
        
# # #         # Add to context with clear section markers
# # #         section_header = f"\n{'='*60}\n{law_name}\nSECTION {section}: {title}\n{'='*60}"
# # #         context_parts.append(f"{section_header}\n{doc.page_content}\n")
        
# # #         # Store source info
# # #         sources.append({
# # #             "section": section,
# # #             "title": title,
# # #             "page": page,
# # #             "score": round(score, 3),
# # #             "law": law_name
# # #         })
        
# # #         if len(sources) >= 6:  # Limit to 6 unique section+law combinations
# # #             break
    
# # #     context_text = "\n".join(context_parts) if context_parts else "No relevant legal sections found in the database."
    
# # #     # Format prompt
# # #     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# # #     prompt = prompt_template.format(
# # #         context=context_text,
# # #         history=history_text,
# # #         question=query
# # #     )
    
# # #     # Generate response
# # #     with st.spinner("💭 Analyzing legal provisions..."):
# # #         try:
# # #             response = model.invoke(prompt).content
# # #         except Exception as e:
# # #             st.error(f"Error generating response: {e}")
# # #             response = "I apologize, but I encountered an error while processing your request. Please try again."
    
# # #     # Display assistant response with sources
# # #     st.session_state.messages.append({
# # #         "role": "assistant",
# # #         "content": response,
# # #         "sources": sources
# # #     })
    
# # #     with st.chat_message("assistant"):
# # #         st.markdown(response)
        
# # #         # Show sources
# # #         if sources:
# # #             with st.expander("📚 View Sources"):
# # #                 for src in sources:
# # #                     relevance_emoji = "🎯" if src['score'] < 0.5 else "✅" if src['score'] < 1.0 else "📄"
# # #                     st.markdown(f"""
# # #                     {relevance_emoji} **{src['law']}**  
# # #                     Section {src['section']}: {src['title']}  
# # #                     Page {src['page']} | Relevance Score: {src['score']:.3f}
# # #                     """)






















# # import streamlit as st
# # from langchain_chroma import Chroma
# # from langchain_groq import ChatGroq
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.prompts import ChatPromptTemplate
# # from dotenv import load_dotenv
# # import os
# # import re

# # # ---------------- Load environment variables ----------------
# # load_dotenv()
# # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # if not GROQ_API_KEY:
# #     st.error("⚠️ GROQ_API_KEY not found! Please set it in your .env file.")
# #     st.stop()

# # # ---------------- Paths ----------------
# # CHROMA_PATH = r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\chroma"

# # # ---------------- Initialize Chroma with matching embeddings ----------------
# # @st.cache_resource
# # def load_database():
# #     hf_embeddings = HuggingFaceEmbeddings(
# #         model_name="sentence-transformers/all-mpnet-base-v2",
# #         model_kwargs={'device': 'cpu'},
# #         encode_kwargs={'normalize_embeddings': True}
# #     )
# #     return Chroma(
# #         persist_directory=CHROMA_PATH,
# #         embedding_function=hf_embeddings,
# #         collection_name="legal_sections"
# #     )

# # db = load_database()

# # # ---------------- Law Detection ----------------
# # def detect_law_context(query):
# #     """Detect which law the user is asking about."""
# #     query_lower = query.lower()
    
# #     # CNS/CNSA keywords (more comprehensive)
# #     cns_keywords = ['cns', 'cnsa', 'narcotic', 'drug', 'opium', 'heroin', 
# #                     'cannabis', 'controlled substance', 'psychotropic', 'hashish',
# #                     'cocaine', 'morphine', 'trafficking', 'cultivation', 'possession',
# #                     'manufacture', 'poppy', 'marijuana', 'methamphetamine']
    
# #     # PPC keywords
# #     ppc_keywords = ['ppc', 'penal code', 'murder', 'theft', 'assault', 
# #                     'culpable homicide', 'hurt', 'wrongful restraint', 'kidnapping',
# #                     'criminal conspiracy', 'rioting', 'defamation', 'cheating']
    
# #     has_cns = any(keyword in query_lower for keyword in cns_keywords)
# #     has_ppc = any(keyword in query_lower for keyword in ppc_keywords)
    
# #     if has_cns and not has_ppc:
# #         return "CNS"
# #     elif has_ppc and not has_cns:
# #         return "PPC"
# #     else:
# #         return "BOTH"  # Return both if ambiguous or both mentioned

# # # ---------------- Section Number Extraction ----------------
# # def extract_section_number(query):
# #     """Extract section number from queries."""
# #     patterns = [
# #         r'\bsection\s+(\d+[A-Z]?)\b',
# #         r'\bsec\.?\s+(\d+[A-Z]?)\b',
# #         r'\bs\.?\s+(\d+[A-Z]?)\b',
# #         r'(?:^|\s)(\d+[A-Z]?)(?:\s|$|[.,!?])',
# #     ]
    
# #     query_lower = query.lower()
# #     for pattern in patterns:
# #         match = re.search(pattern, query_lower)
# #         if match:
# #             return match.group(1).upper()
# #     return None

# # def search_by_section_number(section_num, law_filter=None):
# #     """
# #     Direct metadata search for a specific section number.
# #     If law_filter is specified, only return results from that law.
# #     """
# #     try:
# #         # Get all documents with this section number
# #         results = db.get(where={"section": str(section_num)})
        
# #         if results and results['documents']:
# #             docs_with_scores = []
# #             for i, doc_text in enumerate(results['documents']):
# #                 metadata = results['metadatas'][i] if results['metadatas'] else {}
                
# #                 # Apply law filter if specified
# #                 if law_filter and law_filter != "BOTH":
# #                     law_name = metadata.get('law_name', '')
# #                     if law_filter == "CNS" and "Narcotic" not in law_name:
# #                         continue
# #                     elif law_filter == "PPC" and "Penal Code" not in law_name:
# #                         continue
                
# #                 from langchain.schema import Document
# #                 doc = Document(
# #                     page_content=doc_text,
# #                     metadata=metadata
# #                 )
# #                 docs_with_scores.append((doc, 0.0))
            
# #             return docs_with_scores if docs_with_scores else None
# #     except Exception as e:
# #         st.warning(f"Metadata search failed: {e}")
    
# #     return None

# # # ---------------- Enhanced Prompt Template ----------------
# # PROMPT_TEMPLATE = """You are a knowledgeable legal assistant specializing in Pakistani law, particularly:
# # 1. Pakistan Penal Code (PPC), 1860
# # 2. Control of Narcotic Substances Act (CNSA), 1997

# # Your task is to provide accurate, helpful legal information based ONLY on the provided context.

# # CRITICAL RULES:
# # 1. Answer ONLY using information from the context below - NEVER use external knowledge
# # 2. ALWAYS cite specific sections AND the law name (e.g., "According to Section 9 of the Control of Narcotic Substances Act, 1997...")
# # 3. If a section number appears in multiple laws, mention ALL relevant sections
# # 4. If the exact answer is not in the context, respond: "I apologize, but I don't have information about that specific provision in the available legal documents. Please try rephrasing your question or ask about a different section."
# # 5. Use clear, professional language suitable for educational purposes
# # 6. When referencing sections, include the section number, title, AND law name
# # 7. Quote directly from the section text when appropriate
# # 8. If multiple sections are relevant to the question, mention all of them
# # 9. Provide comprehensive answers that include relevant penalties, definitions, and procedures

# # CONTEXT (Relevant Legal Sections):
# # {context}

# # CONVERSATION HISTORY:
# # {history}

# # QUESTION: {question}

# # ANSWER (with mandatory section and law citations):"""

# # # ---------------- Initialize LLM ----------------
# # @st.cache_resource
# # def load_llm():
# #     return ChatGroq(
# #         api_key=GROQ_API_KEY,
# #         model="llama-3.3-70b-versatile",
# #         temperature=0.1
# #     )

# # model = load_llm()

# # # ---------------- Streamlit UI ----------------
# # st.set_page_config(page_title="ANF EduBot", page_icon="🧑‍⚖️", layout="wide")

# # # Header
# # st.title("🧑‍⚖️ ANF Academy Educational Chatbot")
# # st.caption("Ask questions about Pakistani Law (PPC & CNSA)")

# # # Sidebar with info
# # with st.sidebar:
# #     st.header("📚 About")
# #     st.info("""
# #     This chatbot provides information about:
# #     - **Pakistan Penal Code (PPC), 1860**
# #     - **Control of Narcotic Substances Act (CNSA), 1997**
    
# #     **Tips for best results:**
# #     - Ask about specific sections: "What is Section 9 of CNSA?"
# #     - Ask about concepts: "What are punishments for heroin possession?"
# #     - Ask about offenses: "What are penalties for drug trafficking?"
# #     - Ask about definitions: "What does cultivation mean in CNSA?"
# #     """)
    
# #     st.header("🔍 Database Stats")
# #     try:
# #         collection = db._collection
# #         total = collection.count()
# #         st.metric("Total Chunks", total)
        
# #         # Show breakdown by law
# #         all_docs = db.get()
# #         from collections import Counter
# #         law_counts = Counter(m.get('law_name') for m in all_docs['metadatas'])
        
# #         st.write("**By Law:**")
# #         for law, count in law_counts.items():
# #             st.write(f"• {law}: {count} chunks")
# #     except Exception as e:
# #         st.write("Database loaded ✅")
    
# #     if st.button("🗑️ Clear Chat History"):
# #         st.session_state.messages = []
# #         st.rerun()

# # # Initialize chat history
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # # Display chat history
# # for msg in st.session_state.messages:
# #     with st.chat_message(msg["role"]):
# #         st.markdown(msg["content"])
# #         if "sources" in msg and msg["sources"]:
# #             with st.expander("📚 View Sources"):
# #                 for src in msg["sources"]:
# #                     st.markdown(f"**{src['law']}** - Section {src['section']}: {src['title']} (Page {src['page']})")

# # # User input
# # if query := st.chat_input("Ask a legal question (e.g., 'What are penalties for heroin possession?')"):
# #     # Add user message
# #     st.session_state.messages.append({"role": "user", "content": query})
# #     with st.chat_message("user"):
# #         st.markdown(query)

# #     # Build conversation history (last 10 messages)
# #     history_text = ""
# #     recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
# #     for msg in recent_messages[:-1]:
# #         role = "User" if msg["role"] == "user" else "Assistant"
# #         history_text += f"{role}: {msg['content']}\n"

# #     # HYBRID RETRIEVAL STRATEGY
# #     with st.spinner("🔍 Searching legal documents..."):
# #         results = []
        
# #         # Step 1: Detect law context
# #         law_context = detect_law_context(query)
        
# #         # Step 2: Check if query mentions a specific section number
# #         section_num = extract_section_number(query)
        
# #         if section_num:
# #             st.info(f"🎯 Detected Section {section_num} - Law context: {law_context}")
# #             # Try direct metadata search with law filter
# #             direct_results = search_by_section_number(section_num, law_context)
# #             if direct_results:
# #                 results = direct_results
# #                 st.success(f"✅ Found {len(results)} chunks for Section {section_num}")
        
# #         # Step 3: Always perform semantic search (even if section found, for more context)
# #         if not results or len(results) < 5:
# #             # Build enhanced query based on law context
# #             enhanced_query = query
            
# #             # Add law-specific context to improve search
# #             if law_context == "CNS":
# #                 enhanced_query = f"{query} narcotic drug controlled substance CNSA offense penalty punishment"
# #             elif law_context == "PPC":
# #                 enhanced_query = f"{query} penal code crime offense punishment penalty PPC"
# #             else:
# #                 # For BOTH or ambiguous, let semantic search find the best matches
# #                 enhanced_query = f"{query} law offense penalty punishment section"
            
# #             if section_num:
# #                 enhanced_query = f"{enhanced_query} section {section_num}"
            
# #             # Perform semantic search with higher k value for better coverage
# #             semantic_results = db.similarity_search_with_score(enhanced_query, k=20)
            
# #             # RELAXED filtering - only remove very poor matches
# #             semantic_results = [(doc, score) for doc, score in semantic_results if score < 1.8]
            
# #             # Apply law filter ONLY if explicitly CNS or PPC (not BOTH)
# #             if law_context in ["CNS", "PPC"]:
# #                 filtered_semantic = []
# #                 other_law_results = []
                
# #                 for doc, score in semantic_results:
# #                     law_name = doc.metadata.get('law_name', '')
# #                     if law_context == "CNS" and "Narcotic" in law_name:
# #                         filtered_semantic.append((doc, score))
# #                     elif law_context == "PPC" and "Penal Code" in law_name:
# #                         filtered_semantic.append((doc, score))
# #                     else:
# #                         # Keep other law results as backup
# #                         other_law_results.append((doc, score))
                
# #                 # If we found results from the target law, use them
# #                 if filtered_semantic:
# #                     semantic_results = filtered_semantic
# #                 # Otherwise, keep all results (user might be asking about the other law)
# #                 elif other_law_results:
# #                     semantic_results = other_law_results
            
# #             # Combine direct and semantic results if both exist
# #             if results:
# #                 # Add semantic results that aren't duplicates
# #                 existing_sections = {(doc.metadata.get('law_name'), doc.metadata.get('section')) 
# #                                    for doc, _ in results}
# #                 for doc, score in semantic_results:
# #                     key = (doc.metadata.get('law_name'), doc.metadata.get('section'))
# #                     if key not in existing_sections:
# #                         results.append((doc, score))
# #                         existing_sections.add(key)
# #             else:
# #                 results = semantic_results
            
# #             # Sort results: exact section matches first, then by relevance score
# #             if section_num:
# #                 def sort_key(item):
# #                     doc, score = item
# #                     is_exact = doc.metadata.get("section") == section_num
# #                     return (not is_exact, score)
# #                 results = sorted(results, key=sort_key)
# #             else:
# #                 # Sort by score for semantic queries
# #                 results = sorted(results, key=lambda x: x[1])
        
# #         if len(results) == 0:
# #             st.warning("⚠️ No relevant sections found. Try:\n- Using different keywords\n- Asking about specific offenses or penalties\n- Specifying the law (PPC or CNSA)")
    
# #     # Build context with section references
# #     context_parts = []
# #     sources = []
# #     seen_sections = set()
    
# #     for doc, score in results[:12]:  # Get up to 12 to ensure diversity
# #         section = doc.metadata.get("section", "Unknown")
# #         law_name = doc.metadata.get("law_name", "Unknown Law")
        
# #         # Create unique key combining law and section
# #         section_key = f"{law_name}:{section}"
        
# #         # Skip duplicate section+law combinations
# #         if section_key in seen_sections:
# #             continue
# #         seen_sections.add(section_key)
        
# #         title = doc.metadata.get("title", "")
# #         page = doc.metadata.get("page", "N/A")
        
# #         # Add to context with clear section markers
# #         section_header = f"\n{'='*60}\n{law_name}\nSECTION {section}: {title}\n{'='*60}"
# #         context_parts.append(f"{section_header}\n{doc.page_content}\n")
        
# #         # Store source info
# #         sources.append({
# #             "section": section,
# #             "title": title,
# #             "page": page,
# #             "score": round(score, 3),
# #             "law": law_name
# #         })
        
# #         if len(sources) >= 8:  # Provide more context for semantic queries
# #             break
    
# #     context_text = "\n".join(context_parts) if context_parts else "No relevant legal sections found in the database."
    
# #     # Format prompt
# #     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# #     prompt = prompt_template.format(
# #         context=context_text,
# #         history=history_text,
# #         question=query
# #     )
    
# #     # Generate response
# #     with st.spinner("💭 Analyzing legal provisions..."):
# #         try:
# #             response = model.invoke(prompt).content
# #         except Exception as e:
# #             st.error(f"Error generating response: {e}")
# #             response = "I apologize, but I encountered an error while processing your request. Please try again."
    
# #     # Display assistant response with sources
# #     st.session_state.messages.append({
# #         "role": "assistant",
# #         "content": response,
# #         "sources": sources
# #     })
    
# #     with st.chat_message("assistant"):
# #         st.markdown(response)
        
# #         # Show sources
# #         if sources:
# #             with st.expander("📚 View Sources"):
# #                 for src in sources:
# #                     relevance_emoji = "🎯" if src['score'] < 0.5 else "✅" if src['score'] < 1.0 else "📄"
# #                     st.markdown(f"""
# #                     {relevance_emoji} **{src['law']}**  
# #                     Section {src['section']}: {src['title']}  
# #                     Page {src['page']} | Relevance Score: {src['score']:.3f}
# #                     """)



































# import streamlit as st
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import os
# import re

# # ---------------- Load environment variables ----------------
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     st.error("⚠️ GROQ_API_KEY not found! Please set it in your .env file.")
#     st.stop()

# # ---------------- Paths ----------------
# CHROMA_PATH = r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\chroma"

# # ---------------- Initialize Chroma with matching embeddings ----------------
# @st.cache_resource
# def load_database():
#     hf_embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-mpnet-base-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )
#     return Chroma(
#         persist_directory=CHROMA_PATH,
#         embedding_function=hf_embeddings,
#         collection_name="legal_sections"
#     )

# db = load_database()

# # ---------------- Query Enhancement Mappings ----------------
# QUERY_ENHANCEMENT_MAP = {
#     # Drug-related terms
#     "heroin": "heroin diacetylmorphine opium derivative possession manufacture trafficking section 9",
#     "opium": "opium poppy cultivation possession trafficking morphine section 4 section 9",
#     "cannabis": "cannabis hemp hashish marijuana cultivation possession trafficking section 4 section 9",
#     "cocaine": "cocaine coca leaf coca derivative possession manufacture trafficking section 9",
#     "hashish": "hashish cannabis resin hemp possession trafficking section 9",
#     "drugs": "narcotic drugs psychotropic substances controlled substances possession trafficking",
#     "narcotics": "narcotic drugs opium heroin cannabis cocaine possession manufacture trafficking",
    
#     # Offense-related terms
#     "trafficking": "trafficking transport export import manufacture section 8 section 9 death penalty life imprisonment",
#     "possession": "possession narcotic drug psychotropic substance section 6 section 9 imprisonment penalty",
#     "cultivation": "cultivation cannabis coca opium poppy section 4 section 5 prohibition",
#     "manufacture": "manufacture production narcotic drug psychotropic substance section 10 section 11 premises equipment",
#     "smuggling": "smuggling import export transport trafficking section 7 section 8 section 9",
#     "import": "import export transport narcotic drug section 7 license permit prohibition",
#     "export": "export import transport narcotic drug section 7 license permit prohibition",
    
#     # Penalty-related terms
#     "punishment": "punishment penalty imprisonment fine death sentence section 9 section 11 section 13",
#     "penalty": "penalty punishment imprisonment fine forfeiture section 9 section 18",
#     "death penalty": "death penalty life imprisonment section 9 trafficking manufacture ten kilograms",
#     "imprisonment": "imprisonment punishment penalty rigorous fine section 9 section 11",
#     "fine": "fine penalty punishment imprisonment forfeiture section 9 section 18",
    
#     # Legal procedures
#     "arrest": "arrest warrant search seizure section 20 section 21 section 22 police",
#     "search": "search warrant seizure entry section 20 section 21 premises conveyance",
#     "seizure": "seizure confiscation forfeiture section 21 section 32 narcotic drug",
#     "assets": "assets freezing forfeiture tracing section 12 section 19 section 37 section 39",
#     "forfeiture": "forfeiture confiscation assets property section 13 section 19 section 39",
#     "freezing": "freezing assets forfeiture section 37 section 38 special court",
    
#     # Court-related
#     "bail": "bail section 51 death penalty special court no bail granted",
#     "trial": "trial special court prosecution section 45 section 46 section 47",
#     "appeal": "appeal high court special court section 48 conviction sentence",
#     "special court": "special court jurisdiction trial section 45 section 46 judge sessions",
    
#     # Addiction-related
#     "addict": "addict addiction treatment rehabilitation section 52 section 53 registration",
#     "treatment": "treatment rehabilitation addict de-toxification section 52 section 53 centers",
#     "rehabilitation": "rehabilitation treatment addict section 53 centers provincial government",
    
#     # PPC-related terms
#     "murder": "murder section 302 culpable homicide death punishment qatl",
#     "theft": "theft section 378 section 379 dishonestly movable property",
#     "assault": "assault criminal force section 350 section 351 hurt",
#     "kidnapping": "kidnapping abduction section 359 section 363 section 364",
# }

# def detect_law_context(query):
#     """Detect which law the user is asking about with enhanced detection."""
#     query_lower = query.lower()
    
#     # CNS/CNSA keywords (comprehensive)
#     cns_keywords = [
#         'cns', 'cnsa', 'narcotic', 'drug', 'opium', 'heroin', 'cannabis', 
#         'controlled substance', 'psychotropic', 'hashish', 'cocaine', 'morphine', 
#         'trafficking', 'cultivation', 'possession', 'manufacture', 'poppy', 
#         'marijuana', 'methamphetamine', 'smuggling', 'dealer', 'addict',
#         'hemp', 'charas', 'diacetylmorphine', 'ecgonine', 'alkaloid'
#     ]
    
#     # PPC keywords
#     ppc_keywords = [
#         'ppc', 'penal code', 'murder', 'theft', 'assault', 'culpable homicide', 
#         'hurt', 'wrongful restraint', 'kidnapping', 'criminal conspiracy', 
#         'rioting', 'defamation', 'cheating', 'qatl', 'grievous hurt'
#     ]
    
#     cns_count = sum(1 for keyword in cns_keywords if keyword in query_lower)
#     ppc_count = sum(1 for keyword in ppc_keywords if keyword in query_lower)
    
#     if cns_count > ppc_count and cns_count > 0:
#         return "CNS"
#     elif ppc_count > cns_count and ppc_count > 0:
#         return "PPC"
#     else:
#         return "BOTH"

# def enhance_query_with_context(query):
#     """
#     Enhance query by adding relevant legal terms and context.
#     This helps capture generic questions better.
#     """
#     query_lower = query.lower()
#     enhancements = []
    
#     # Find matching enhancement terms
#     for term, enhancement in QUERY_ENHANCEMENT_MAP.items():
#         if term in query_lower:
#             enhancements.append(enhancement)
    
#     # If enhancements found, combine with original query
#     if enhancements:
#         enhanced = f"{query} {' '.join(enhancements)}"
#         return enhanced
    
#     return query

# def extract_section_number(query):
#     """Extract section number from queries with enhanced patterns."""
#     patterns = [
#         r'\bsection\s+(\d+[A-Z]?)\b',
#         r'\bsec\.?\s+(\d+[A-Z]?)\b',
#         r'\bs\.?\s+(\d+[A-Z]?)\b',
#         r'(?:^|\s)(\d+[A-Z]?)(?:\s+of|\s+in|\s|$|[.,!?])',
#     ]
    
#     query_lower = query.lower()
#     for pattern in patterns:
#         match = re.search(pattern, query_lower)
#         if match:
#             return match.group(1).upper()
#     return None

# def search_by_section_number(section_num, law_filter=None):
#     """Direct metadata search for a specific section number."""
#     try:
#         results = db.get(where={"section": str(section_num)})
        
#         if results and results['documents']:
#             docs_with_scores = []
#             for i, doc_text in enumerate(results['documents']):
#                 metadata = results['metadatas'][i] if results['metadatas'] else {}
                
#                 if law_filter and law_filter != "BOTH":
#                     law_name = metadata.get('law_name', '')
#                     if law_filter == "CNS" and "Narcotic" not in law_name:
#                         continue
#                     elif law_filter == "PPC" and "Penal Code" not in law_name:
#                         continue
                
#                 from langchain.schema import Document
#                 doc = Document(page_content=doc_text, metadata=metadata)
#                 docs_with_scores.append((doc, 0.0))
            
#             return docs_with_scores if docs_with_scores else None
#     except Exception as e:
#         st.warning(f"Metadata search failed: {e}")
    
#     return None

# def multi_strategy_search(query, law_context, section_num=None):
#     """
#     Multi-strategy search that combines:
#     1. Direct section lookup
#     2. Enhanced semantic search
#     3. Multi-query search with variations
#     4. Keyword-based filtering
#     """
#     all_results = []
#     seen_keys = set()
    
#     # Strategy 1: Direct section search if section number detected
#     if section_num:
#         direct_results = search_by_section_number(section_num, law_context)
#         if direct_results:
#             for doc, score in direct_results:
#                 key = (doc.metadata.get('law_name'), doc.metadata.get('section'))
#                 if key not in seen_keys:
#                     all_results.append((doc, score, "direct"))
#                     seen_keys.add(key)
    
#     # Strategy 2: Enhanced semantic search with query expansion
#     enhanced_query = enhance_query_with_context(query)
    
#     # Add law-specific boosting
#     if law_context == "CNS":
#         enhanced_query = f"{enhanced_query} CNSA narcotic drug offense penalty section"
#     elif law_context == "PPC":
#         enhanced_query = f"{enhanced_query} PPC penal code crime offense section"
    
#     semantic_results = db.similarity_search_with_score(enhanced_query, k=25)
    
#     for doc, score in semantic_results:
#         key = (doc.metadata.get('law_name'), doc.metadata.get('section'))
#         if key not in seen_keys and score < 1.9:
#             all_results.append((doc, score, "semantic"))
#             seen_keys.add(key)
    
#     # Strategy 3: Multi-query variations for better coverage
#     query_variations = generate_query_variations(query, law_context)
    
#     for variation in query_variations:
#         var_results = db.similarity_search_with_score(variation, k=10)
#         for doc, score in var_results:
#             key = (doc.metadata.get('law_name'), doc.metadata.get('section'))
#             if key not in seen_keys and score < 1.7:
#                 all_results.append((doc, score, "variation"))
#                 seen_keys.add(key)
    
#     # Strategy 4: Filter by law context
#     if law_context in ["CNS", "PPC"]:
#         filtered_results = []
#         for doc, score, source in all_results:
#             law_name = doc.metadata.get('law_name', '')
#             if law_context == "CNS" and "Narcotic" in law_name:
#                 filtered_results.append((doc, score, source))
#             elif law_context == "PPC" and "Penal Code" in law_name:
#                 filtered_results.append((doc, score, source))
#             elif law_context == "BOTH":
#                 filtered_results.append((doc, score, source))
        
#         if filtered_results:
#             all_results = filtered_results
    
#     # Sort by score and source priority
#     def sort_key(item):
#         doc, score, source = item
#         source_priority = {"direct": 0, "semantic": 1, "variation": 2}
#         is_exact_section = doc.metadata.get("section") == section_num if section_num else False
#         return (not is_exact_section, source_priority.get(source, 3), score)
    
#     all_results = sorted(all_results, key=sort_key)
    
#     # Return top results (doc, score tuples)
#     return [(doc, score) for doc, score, _ in all_results[:15]]

# def generate_query_variations(query, law_context):
#     """Generate query variations for better search coverage."""
#     variations = []
#     query_lower = query.lower()
    
#     # Variation 1: Add "what is" prefix
#     if not query_lower.startswith(("what", "how", "why", "when", "where")):
#         variations.append(f"what is {query}")
    
#     # Variation 2: Add law-specific context
#     if law_context == "CNS":
#         variations.append(f"{query} under CNSA narcotic substances act")
#         variations.append(f"penalties for {query} drugs narcotics")
#     elif law_context == "PPC":
#         variations.append(f"{query} under PPC pakistan penal code")
#         variations.append(f"punishment for {query} criminal offense")
    
#     # Variation 3: Rephrase common questions
#     if "punishment" in query_lower or "penalty" in query_lower:
#         variations.append(f"what are the consequences of {query}")
#         variations.append(f"imprisonment fine for {query}")
    
#     if "what is" in query_lower:
#         variations.append(query_lower.replace("what is", "definition of"))
#         variations.append(query_lower.replace("what is", "explain"))
    
#     return variations[:3]  # Limit to 3 variations to avoid too many queries

# # ---------------- Enhanced Prompt Template ----------------
# PROMPT_TEMPLATE = """You are a knowledgeable legal assistant specializing in Pakistani law, particularly:
# 1. Pakistan Penal Code (PPC), 1860
# 2. Control of Narcotic Substances Act (CNSA), 1997

# Your task is to provide accurate, helpful legal information based ONLY on the provided context.

# CRITICAL RULES:
# 1. Answer ONLY using information from the context below - NEVER use external knowledge
# 2. ALWAYS cite specific sections AND the law name (e.g., "According to Section 9 of the Control of Narcotic Substances Act, 1997...")
# 3. If a section number appears in multiple laws, mention ALL relevant sections
# 4. If the exact answer is not in the context, respond: "I apologize, but I don't have information about that specific provision in the available legal documents. Please try rephrasing your question or ask about a different section."
# 5. Use clear, professional language suitable for educational purposes
# 6. When referencing sections, include the section number, title, AND law name
# 7. Quote directly from the section text when appropriate
# 8. If multiple sections are relevant to the question, mention all of them
# 9. Provide comprehensive answers that include relevant penalties, definitions, and procedures
# 10. Structure your answer with clear paragraphs and bullet points where appropriate

# CONTEXT (Relevant Legal Sections):
# {context}

# CONVERSATION HISTORY:
# {history}

# QUESTION: {question}

# ANSWER (with mandatory section and law citations):"""

# # ---------------- Initialize LLM ----------------
# @st.cache_resource
# def load_llm():
#     return ChatGroq(
#         api_key=GROQ_API_KEY,
#         model="llama-3.3-70b-versatile",
#         temperature=0.1
#     )

# model = load_llm()

# # ---------------- Streamlit UI ----------------
# st.set_page_config(page_title="ANF EduBot", page_icon="🧑‍⚖️", layout="wide")

# st.title("🧑‍⚖️ ANF Academy Educational Chatbot")
# st.caption("Ask questions about Pakistani Law (PPC & CNSA)")

# with st.sidebar:
#     st.header("📚 About")
#     st.info("""
#     This chatbot provides information about:
#     - **Pakistan Penal Code (PPC), 1860**
#     - **Control of Narcotic Substances Act (CNSA), 1997**
    
#     **Tips for best results:**
#     - Ask about specific sections: "What is Section 9 of CNSA?"
#     - Ask about concepts: "What are punishments for heroin possession?"
#     - Ask about offenses: "What are penalties for drug trafficking?"
#     - Ask about definitions: "What does cultivation mean in CNSA?"
#     - Ask generic questions: "Tell me about drug-related offenses"
#     """)
    
#     st.header("🔍 Database Stats")
#     try:
#         collection = db._collection
#         total = collection.count()
#         st.metric("Total Chunks", total)
        
#         all_docs = db.get()
#         from collections import Counter
#         law_counts = Counter(m.get('law_name') for m in all_docs['metadatas'])
        
#         st.write("**By Law:**")
#         for law, count in law_counts.items():
#             st.write(f"• {law}: {count} chunks")
#     except Exception as e:
#         st.write("Database loaded ✅")
    
#     if st.button("🗑️ Clear Chat History"):
#         st.session_state.messages = []
#         st.rerun()

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if "sources" in msg and msg["sources"]:
#             with st.expander("📚 View Sources"):
#                 for src in msg["sources"]:
#                     st.markdown(f"**{src['law']}** - Section {src['section']}: {src['title']}")

# if query := st.chat_input("Ask a legal question (e.g., 'What are penalties for heroin possession?')"):
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     history_text = ""
#     recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
#     for msg in recent_messages[:-1]:
#         role = "User" if msg["role"] == "user" else "Assistant"
#         history_text += f"{role}: {msg['content']}\n"

#     with st.spinner("🔍 Searching legal documents..."):
#         law_context = detect_law_context(query)
#         section_num = extract_section_number(query)
        
#         if section_num:
#             st.info(f"🎯 Detected Section {section_num} - Law context: {law_context}")
        
#         # Use multi-strategy search
#         results = multi_strategy_search(query, law_context, section_num)
        
#         if results:
#             st.success(f"✅ Found {len(results)} relevant sections")
#         else:
#             st.warning("⚠️ No relevant sections found. Try rephrasing your question.")
    
#     context_parts = []
#     sources = []
#     seen_sections = set()
    
#     for doc, score in results[:10]:
#         section = doc.metadata.get("section", "Unknown")
#         law_name = doc.metadata.get("law_name", "Unknown Law")
#         section_key = f"{law_name}:{section}"
        
#         if section_key in seen_sections:
#             continue
#         seen_sections.add(section_key)
        
#         title = doc.metadata.get("title", "")
#         page = doc.metadata.get("page", "N/A")
        
#         section_header = f"\n{'='*60}\n{law_name}\nSECTION {section}: {title}\n{'='*60}"
#         context_parts.append(f"{section_header}\n{doc.page_content}\n")
        
#         sources.append({
#             "section": section,
#             "title": title,
#             "page": page,
#             "score": round(score, 3),
#             "law": law_name
#         })
        
#         if len(sources) >= 8:
#             break
    
#     context_text = "\n".join(context_parts) if context_parts else "No relevant legal sections found in the database."
    
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(
#         context=context_text,
#         history=history_text,
#         question=query
#     )
    
#     with st.spinner("💭 Analyzing legal provisions..."):
#         try:
#             response = model.invoke(prompt).content
#         except Exception as e:
#             st.error(f"Error generating response: {e}")
#             response = "I apologize, but I encountered an error while processing your request. Please try again."
    
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": response,
#         "sources": sources
#     })
    
#     with st.chat_message("assistant"):
#         st.markdown(response)
        
#         if sources:
#             with st.expander("📚 View Sources"):
#                 for src in sources:
#                     relevance_emoji = "🎯" if src['score'] < 0.5 else "✅" if src['score'] < 1.0 else "📄"
#                     st.markdown(f"""
#                     {relevance_emoji} **{src['law']}**  
#                     Section {src['section']}: {src['title']}  
#                     Page {src['page']} | Relevance Score: {src['score']:.3f}
#                     """)



































import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import re





# ---------------- Load environment variables ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found! Please set it in your .env file.")
    st.stop()

# ---------------- Paths ----------------
CHROMA_PATH = r"D:\UMER_ANF\Edubot_old_laptop\chroma"

# ---------------- Initialize Chroma with matching embeddings ----------------
@st.cache_resource
def load_database():
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=hf_embeddings,
        collection_name="legal_sections"
    )

db = load_database()

# ---------------- Enhanced Query Enhancement Mappings ----------------
QUERY_ENHANCEMENT_MAP = {
    # Drug-related terms
    "heroin": "heroin diacetylmorphine opium derivative possession manufacture trafficking section 9",
    "opium": "opium poppy cultivation possession trafficking morphine section 4 section 9",
    "cannabis": "cannabis hemp hashish marijuana cultivation possession trafficking section 4 section 9",
    "cocaine": "cocaine coca leaf coca derivative possession manufacture trafficking section 9",
    "hashish": "hashish cannabis resin hemp possession trafficking section 9",
    "drugs": "narcotic drugs psychotropic substances controlled substances possession trafficking",
    "narcotics": "narcotic drugs opium heroin cannabis cocaine possession manufacture trafficking",
    
    # Offense-related terms
    "trafficking": "trafficking transport export import manufacture section 8 section 9 death penalty life imprisonment",
    "possession": "possession narcotic drug psychotropic substance section 6 section 9 imprisonment penalty",
    "cultivation": "cultivation cannabis coca opium poppy section 4 section 5 prohibition",
    "manufacture": "manufacture production narcotic drug psychotropic substance section 10 section 11 premises equipment",
    "smuggling": "smuggling import export transport trafficking section 7 section 8 section 9",
    "import": "import export transport narcotic drug section 7 license permit prohibition",
    "export": "export import transport narcotic drug section 7 license permit prohibition",
    
    # Penalty-related terms
    "punishment": "punishment penalty imprisonment fine death sentence section 9 section 11 section 13",
    "penalty": "penalty punishment imprisonment fine forfeiture section 9 section 18",
    "death penalty": "death penalty life imprisonment section 9 trafficking manufacture ten kilograms",
    "imprisonment": "imprisonment punishment penalty rigorous fine section 9 section 11",
    "fine": "fine penalty punishment imprisonment forfeiture section 9 section 18",
    
    # Legal procedures
    "arrest": "arrest warrant search seizure section 20 section 21 section 22 police",
    "search": "search warrant seizure entry section 20 section 21 premises conveyance",
    "seizure": "seizure confiscation forfeiture section 21 section 32 narcotic drug",
    "assets": "assets freezing forfeiture tracing section 12 section 19 section 37 section 39",
    "forfeiture": "forfeiture confiscation assets property section 13 section 19 section 39",
    "freezing": "freezing assets forfeiture section 37 section 38 special court",
    
    # Court-related
    "bail": "bail section 51 death penalty special court no bail granted",
    "trial": "trial special court prosecution section 45 section 46 section 47",
    "appeal": "appeal high court special court section 48 conviction sentence",
    "special court": "special court jurisdiction trial section 45 section 46 judge sessions",
    
    # Addiction-related
    "addict": "addict addiction treatment rehabilitation section 52 section 53 registration",
    "treatment": "treatment rehabilitation addict de-toxification section 52 section 53 centers",
    "rehabilitation": "rehabilitation treatment addict section 53 centers provincial government",
    
    # PPC-related terms
    "murder": "murder section 302 culpable homicide death punishment qatl",
    "theft": "theft section 378 section 379 dishonestly movable property",
    "assault": "assault criminal force section 350 section 351 hurt",
    "kidnapping": "kidnapping abduction section 359 section 363 section 364",
    
    # Police Rules related terms
    "jail": "jail prison prisoner gazetted officer subordinate interrogate",
    "jails": "jails prison prisoners gazetted officer subordinate interrogate entry powers",
    "enter": "enter entry powers premises search warrant",
    "powers": "powers authority officer police duty discharge",
    "police rules": "police rules Punjab gazette officer subordinate duty",
    "police officer": "police officer gazetted subordinate inspector powers duty",
}

# ---------------- Enhanced Law Detection ----------------
def detect_law_context(query):
    """Detect which law the user is asking about with enhanced detection."""
    query_lower = query.lower()
    
    # CNS/CNSA keywords
    cns_keywords = [
        'cns', 'cnsa', 'narcotic', 'drug', 'opium', 'heroin', 'cannabis', 
        'controlled substance', 'psychotropic', 'hashish', 'cocaine', 'morphine', 
        'trafficking', 'cultivation', 'possession', 'manufacture', 'poppy', 
        'marijuana', 'methamphetamine', 'smuggling', 'dealer', 'addict',
        'hemp', 'charas', 'diacetylmorphine', 'ecgonine', 'alkaloid'
    ]
    
    # PPC keywords
    ppc_keywords = [
        'ppc', 'penal code', 'murder', 'theft', 'assault', 'culpable homicide', 
        'hurt', 'wrongful restraint', 'kidnapping', 'criminal conspiracy', 
        'rioting', 'defamation', 'cheating', 'qatl', 'grievous hurt'
    ]
    
    # Police Rules keywords
    police_keywords = [
        'police rules', 'punjab police', 'police officer', 'gazetted', 
        'subordinate', 'jail', 'prison', 'patrol', 'duty', 'uniform'
    ]
    
    # AMLA keywords
    amla_keywords = [
        'amla', 'money laundering', 'anti-money laundering', 'proceeds of crime',
        'suspicious transaction', 'financial monitoring'
    ]
    
    # ANF Act keywords
    anf_keywords = [
        'anf act', 'anti narcotics force', 'anf officer', 'anf operations'
    ]
    
    # Qanun-e-Shahadat keywords
    qes_keywords = [
        'qanun-e-shahadat', 'evidence', 'testimony', 'witness', 'proof'
    ]
    
    cns_count = sum(1 for keyword in cns_keywords if keyword in query_lower)
    ppc_count = sum(1 for keyword in ppc_keywords if keyword in query_lower)
    police_count = sum(1 for keyword in police_keywords if keyword in query_lower)
    amla_count = sum(1 for keyword in amla_keywords if keyword in query_lower)
    anf_count = sum(1 for keyword in anf_keywords if keyword in query_lower)
    qes_count = sum(1 for keyword in qes_keywords if keyword in query_lower)
    
    # Return the law with highest match count
    counts = {
        "CNS": cns_count,
        "PPC": ppc_count,
        "POLICE": police_count,
        "AMLA": amla_count,
        "ANF": anf_count,
        "QES": qes_count
    }
    
    max_count = max(counts.values())
    if max_count == 0:
        return "BOTH"
    
    # Return the law(s) with highest count
    top_laws = [law for law, count in counts.items() if count == max_count]
    return top_laws[0] if len(top_laws) == 1 else "BOTH"

# ---------------- Enhanced Section Number Extraction ----------------
def extract_section_number(query):
    """Extract section number from queries with support for decimal formats."""
    patterns = [
        # Decimal formats like 14.21, 1.1, 10.132
        r'\bsection\s+(\d+\.\d+[A-Z]?)\b',
        r'\bsec\.?\s+(\d+\.\d+[A-Z]?)\b',
        r'\bs\.?\s+(\d+\.\d+[A-Z]?)\b',
        r'(?:^|\s)(\d+\.\d+[A-Z]?)(?:\s|$|[.,!?])',
        
        # Standard formats
        r'\bsection\s+(\d+[A-Z]?)\b',
        r'\bsec\.?\s+(\d+[A-Z]?)\b',
        r'\bs\.?\s+(\d+[A-Z]?)\b',
        
        # Rule formats
        r'\brule\s+(\d+\.\d+[A-Z]?)\b',
        r'\brule\s+(\d+[A-Z]?)\b',
        
        # Article formats
        r'\barticle\s+(\d+\.\d+[A-Z]?)\b',
        r'\barticle\s+(\d+[A-Z]?)\b',
    ]
    
    query_lower = query.lower()
    for pattern in patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None

def search_by_section_number(section_num, law_filter=None):
    """
    Direct metadata search for a specific section number.
    Enhanced to handle decimal section numbers.
    """
    try:
        # Try exact match first
        results = db.get(where={"section": str(section_num)})
        
        # If no exact match, try case-insensitive and with/without leading zeros
        if not results or not results['documents']:
            # Get all documents and filter manually
            all_results = db.get()
            matching_indices = []
            
            for i, metadata in enumerate(all_results.get('metadatas', [])):
                stored_section = str(metadata.get('section', '')).upper().strip()
                search_section = str(section_num).upper().strip()
                
                # Check for match (exact or normalized)
                if stored_section == search_section:
                    matching_indices.append(i)
                # Also check if removing leading zeros matches
                elif stored_section.lstrip('0') == search_section.lstrip('0'):
                    matching_indices.append(i)
            
            if matching_indices:
                results = {
                    'documents': [all_results['documents'][i] for i in matching_indices],
                    'metadatas': [all_results['metadatas'][i] for i in matching_indices]
                }
        
        if results and results['documents']:
            docs_with_scores = []
            for i, doc_text in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                
                # Apply law filter if specified
                if law_filter and law_filter not in ["BOTH", "ALL"]:
                    law_name = metadata.get('law_name', '').lower()
                    
                    if law_filter == "CNS" and "narcotic" not in law_name:
                        continue
                    elif law_filter == "PPC" and "penal code" not in law_name:
                        continue
                    elif law_filter == "POLICE" and "police rules" not in law_name:
                        continue
                    elif law_filter == "AMLA" and "money laundering" not in law_name:
                        continue
                    elif law_filter == "ANF" and "anf act" not in law_name:
                        continue
                    elif law_filter == "QES" and "shahadat" not in law_name:
                        continue
                
                from langchain.schema import Document
                doc = Document(page_content=doc_text, metadata=metadata)
                docs_with_scores.append((doc, 0.0))
            
            return docs_with_scores if docs_with_scores else None
    except Exception as e:
        st.warning(f"Metadata search failed: {e}")
    
    return None

def enhance_query_with_context(query):
    """Enhanced query with better context and law-specific terms."""
    query_lower = query.lower()
    enhancements = []
    
    # Find matching enhancement terms
    for term, enhancement in QUERY_ENHANCEMENT_MAP.items():
        if term in query_lower:
            enhancements.append(enhancement)
    
    # Add law-specific enhancements based on detected context
    if "police" in query_lower or "jail" in query_lower:
        enhancements.append("Punjab Police Rules 1934 gazetted officer subordinate")
    
    if "section" in query_lower or "rule" in query_lower:
        enhancements.append("provisions law statute regulation")
    
    # If enhancements found, combine with original query
    if enhancements:
        enhanced = f"{query} {' '.join(enhancements)}"
        return enhanced
    
    return query

def multi_strategy_search(query, law_context, section_num=None):
    """
    Enhanced multi-strategy search with better handling of all laws.
    """
    all_results = []
    seen_keys = set()
    
    # Strategy 1: Direct section search if section number detected
    if section_num:
        st.info(f"🎯 Searching for Section {section_num} in {law_context}")
        direct_results = search_by_section_number(section_num, law_context)
        if direct_results:
            st.success(f"✅ Found {len(direct_results)} direct matches for Section {section_num}")
            for doc, score in direct_results:
                key = (doc.metadata.get('law_name'), doc.metadata.get('section'))
                if key not in seen_keys:
                    all_results.append((doc, score, "direct"))
                    seen_keys.add(key)
    
    # Strategy 2: Enhanced semantic search
    enhanced_query = enhance_query_with_context(query)
    
    # Add law-specific boosting
    if law_context == "CNS":
        enhanced_query = f"{enhanced_query} CNSA narcotic drug offense penalty section"
    elif law_context == "PPC":
        enhanced_query = f"{enhanced_query} PPC penal code crime offense section"
    elif law_context == "POLICE":
        enhanced_query = f"{enhanced_query} Punjab Police Rules officer duty powers"
    elif law_context == "AMLA":
        enhanced_query = f"{enhanced_query} AMLA money laundering proceeds crime"
    elif law_context == "ANF":
        enhanced_query = f"{enhanced_query} ANF Act narcotics force operations"
    elif law_context == "QES":
        enhanced_query = f"{enhanced_query} Qanun-e-Shahadat evidence testimony proof"
    
    semantic_results = db.similarity_search_with_score(enhanced_query, k=30)
    
    for doc, score in semantic_results:
        key = (doc.metadata.get('law_name'), doc.metadata.get('section'))
        if key not in seen_keys and score < 2.0:
            all_results.append((doc, score, "semantic"))
            seen_keys.add(key)
    
    # Strategy 3: Multi-query variations
    query_variations = generate_query_variations(query, law_context, section_num)
    
    for variation in query_variations:
        var_results = db.similarity_search_with_score(variation, k=15)
        for doc, score in var_results:
            key = (doc.metadata.get('law_name'), doc.metadata.get('section'))
            if key not in seen_keys and score < 1.8:
                all_results.append((doc, score, "variation"))
                seen_keys.add(key)
    
    # Strategy 4: Filter by law context (more flexible)
    if law_context not in ["BOTH", "ALL"]:
        filtered_results = []
        for doc, score, source in all_results:
            law_name = doc.metadata.get('law_name', '').lower()
            
            if law_context == "CNS" and "narcotic" in law_name:
                filtered_results.append((doc, score, source))
            elif law_context == "PPC" and "penal code" in law_name:
                filtered_results.append((doc, score, source))
            elif law_context == "POLICE" and "police rules" in law_name:
                filtered_results.append((doc, score, source))
            elif law_context == "AMLA" and "money laundering" in law_name:
                filtered_results.append((doc, score, source))
            elif law_context == "ANF" and "anf act" in law_name:
                filtered_results.append((doc, score, source))
            elif law_context == "QES" and "shahadat" in law_name:
                filtered_results.append((doc, score, source))
        
        if filtered_results:
            all_results = filtered_results
    
    # Sort by score and source priority
    def sort_key(item):
        doc, score, source = item
        source_priority = {"direct": 0, "semantic": 1, "variation": 2}
        is_exact_section = doc.metadata.get("section") == section_num if section_num else False
        return (not is_exact_section, source_priority.get(source, 3), score)
    
    all_results = sorted(all_results, key=sort_key)
    
    # Return top results
    return [(doc, score) for doc, score, _ in all_results[:20]]

def generate_query_variations(query, law_context, section_num=None):
    """Generate query variations for better search coverage."""
    variations = []
    query_lower = query.lower()
    
    # Remove filler words for more focused search
    cleaned_query = re.sub(r'\b(tell me about|what is|explain|describe)\b', '', query_lower).strip()
    if cleaned_query != query_lower:
        variations.append(cleaned_query)
    
    # Add section-focused variations
    if section_num:
        variations.append(f"section {section_num} provisions powers duties")
        variations.append(f"section {section_num} text content")
    
    # Add law-specific variations
    if law_context == "POLICE":
        variations.append(f"{cleaned_query} Punjab Police Rules officer duties")
        variations.append(f"{cleaned_query} police powers authority")
    
    # Add title-based variations
    variations.append(f"title heading {cleaned_query}")
    
    return variations[:5]

# ---------------- Enhanced Prompt Template ----------------
PROMPT_TEMPLATE = """You are a knowledgeable legal assistant specializing in Pakistani law, including:
1. Pakistan Penal Code (PPC), 1860
2. Control of Narcotic Substances Act (CNSA), 1997
3. Punjab Police Rules, 1934
4. Anti-Money Laundering Act (AMLA), 2010
5. Anti Narcotics Force Act, 1997
6. Qanun-e-Shahadat Order, 1984

CRITICAL RULES:
1. Answer ONLY using information from the context below
2. ALWAYS cite specific sections with the FULL law name
3. If multiple sections are relevant, mention ALL of them
4. Quote directly from the text when appropriate
5. If information is not in context, say: "I don't have information about that specific provision in the available documents."
6. Use clear, professional language
7. Structure answers with paragraphs and bullet points where helpful

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {question}

ANSWER:"""

# ---------------- Initialize LLM ----------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

model = load_llm()

# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="ANF EduBot", page_icon="🧑‍⚖️", layout="wide")

st.title("🧑‍⚖️ ANF Academy Educational Chatbot")
st.caption("Ask questions about Pakistani Law (PPC, CNSA, Police Rules, AMLA, ANF Act, Qanun-e-Shahadat)")

with st.sidebar:
    st.header("📚 About")
    st.info("""
    **Tips for asking questions:**
  
    
    • **Understanding Crimes:** I keep hearing about 'money laundering' on the news. Can you explain it simply?
    
    • **Authority Questions:** If an ANF officer stops my car, what are they allowed to do?
    
    • **Process Questions:** How does a bank report someone they think is doing something shady with their money?
    
    • **Consequence Questions:** Can you get the death penalty for drugs in Pakistan?
    
    • **Comparison Questions:** What's the difference between 'charas' and 'heroin' under the law?
    
    Feel free to ask in your own words!

    
    **This chatbot covers:**
    - Pakistan Penal Code (PPC)
    - Control of Narcotic Substances Act (CNSA)
    - Punjab Police Rules, 1934
    - Anti-Money Laundering Act (AMLA)
    - Anti Narcotics Force Act
    - Qanun-e-Shahadat Order
    
    
    """)
    
    st.header("🔍 Database Stats")
    try:
        count = db._collection.count()
        st.metric("Total Chunks", count)
        
        all_docs = db.get()
        from collections import Counter
        law_counts = Counter(m.get('law_name') for m in all_docs['metadatas'])
        
        st.write("**By Law:**")
        for law, cnt in law_counts.items():
            st.write(f"• {law}: {cnt}")
    except:
        st.write("Database loaded ✅")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['law']}** - Sec {src['section']}: {src['title']}")

if query := st.chat_input("Ask a legal question"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history_text = "\n".join([
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in st.session_state.messages[-10:-1]
    ])

    with st.spinner("🔍 Searching..."):
        law_context = detect_law_context(query)
        section_num = extract_section_number(query)
        
        results = multi_strategy_search(query, law_context, section_num)
        
        if results:
            st.success(f"✅ Found {len(results)} relevant sections")
        else:
            st.warning("⚠️ No results. Try different keywords.")
    
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
        page = doc.metadata.get("page", "N/A")
        
        header = f"\n{'='*60}\n{law}\nSECTION {sec}: {title}\n{'='*60}"
        context_parts.append(f"{header}\n{doc.page_content}\n")
        
        sources.append({
            "section": sec,
            "title": title,
            "page": page,
            "score": round(score, 3),
            "law": law
        })
        
        if len(sources) >= 10:
            break
    
    context_text = "\n".join(context_parts) if context_parts else "No relevant sections found."
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        history=history_text,
        question=query
    )
    
    with st.spinner("💭 Analyzing..."):
        try:
            response = model.invoke(prompt).content
        except Exception as e:
            st.error(f"Error: {e}")
            response = "Error processing request. Please try again."
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
    
    with st.chat_message("assistant"):
        st.markdown(response)
        
        if sources:
            with st.expander("📚 Sources"):
                for src in sources:
                    emoji = "🎯" if src['score'] < 0.5 else "✅" if src['score'] < 1.0 else "📄"
                    st.markdown(f"""
                    {emoji} **{src['law']}**  
                    Section {src['section']}: {src['title']}  
                    Page {src['page']} | Score: {src['score']:.3f}
                    """)