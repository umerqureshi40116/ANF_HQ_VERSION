import json
import os
import shutil
import re
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class LegalDocumentIndexer:
    def __init__(self, json_files: List[str], chroma_path: str, collection_name: str = "legal_sections"):
        self.json_files = json_files
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.documents = []
    
    @staticmethod
    def clean_legal_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = " ".join(text.split())
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)
        return text.strip()
    
    @staticmethod
    def extract_section_number(raw_sec: str) -> str:
        raw_sec = str(raw_sec).strip()
        match = re.match(r"(?:Section|Article|Rule)?\s*([\dA-Za-z.\-() ]+)", raw_sec)
        return match.group(1).strip() if match else raw_sec
    
    @staticmethod
    def extract_sections_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sections = []
        if "sections" in data:
            sections.extend(data["sections"])
        if "volumes" in data and isinstance(data["volumes"], dict):
            for vol_name, vol_sections in data["volumes"].items():
                for sec in vol_sections:
                    sec["volume"] = vol_name
                    sections.append(sec)
        return sections
    
    def load_json_files(self) -> None:
        print("\n[LOADING] JSON Files")
        print("=" * 60)
        
        for file_path in self.json_files:
            if not os.path.exists(file_path):
                print(f"SKIP: {file_path} not found")
                continue
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            law_name = data.get("law_name", "Unknown Law")
            sections = self.extract_sections_from_json(data)
            
            print(f"LOAD: {law_name} - {len(sections)} sections")
            
            for sec in sections:
                raw_sec = str(sec.get("section", ""))
                section_num = self.extract_section_number(raw_sec)
                
                title = self.clean_legal_text(sec.get("title", ""))
                body = self.clean_legal_text(sec.get("body", ""))
                
                if len(body) < 20:
                    continue
                
                section_label = f"Section {section_num}".strip()
                if "Rule" in raw_sec or "rule" in raw_sec.lower():
                    section_label = f"Rule {section_num}"
                elif "Article" in raw_sec or "article" in raw_sec.lower():
                    section_label = f"Article {section_num}"
                
                full_content = f"{section_label}: {title}\n\n{body}"
                
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "section": section_num,
                        "title": title,
                        "body": body,
                        "chapter": sec.get("chapter", "") or sec.get("part", ""),
                        "volume": sec.get("volume", ""),
                        "page": int(sec.get("page", -1)) if sec.get("page") else -1,
                        "law_name": law_name,
                        "source": law_name
                    }
                )
                self.documents.append(doc)
        
        print(f"\nTOTAL: {len(self.documents)} documents loaded")
    
    def chunk_documents(self, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
        print("\n[CHUNKING] Documents")
        print("=" * 60)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(self.documents)
        print(f"CHUNKS: {len(chunks)} chunks created")
        return chunks
    
    def index_to_chroma(self, chunks: List[Document]) -> None:
        print("\n[INDEXING] Chroma Database")
        print("=" * 60)
        
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print("RESET: Removed old database")
        
        print("INIT: Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print(f"SAVE: Indexing {len(chunks)} chunks...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.chroma_path,
            collection_name=self.collection_name
        )
        
        print(f"DONE: Database saved to {self.chroma_path}")
        self._verify_index(embeddings)
    
    def _verify_index(self, embeddings) -> None:
        print("\n[VERIFY] Index Integrity")
        print("=" * 60)
        
        db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=embeddings,
            collection_name=self.collection_name
        )
        
        count = db._collection.count()
        print(f"VERIFY: {count} chunks indexed")
        
        if count > 0:
            docs = db.get(limit=5)
            print("\nSAMPLE METADATA:")
            for meta in docs.get("metadatas", []):
                law = meta.get("law_name", "Unknown")
                section = meta.get("section", "Unknown")
                print(f"  {law} - Section {section}")
        else:
            print("WARNING: No chunks found in database")


def main():
    json_files = [
        r"D:\UMER_ANF\Edubot_old_laptop\data\books\ppc_full_sections_left_superscript.json",
        r"D:\UMER_ANF\Edubot_old_laptop\data\books\cnsa_sections_extracted.json",
        r"D:\UMER_ANF\Edubot_old_laptop\data\books\amla_2010_final_structured.json",
        r"D:\UMER_ANF\Edubot_old_laptop\data\books\ANF_ACT_1997.json",
        r"D:\UMER_ANF\Edubot_old_laptop\data\books\punjab_police_rules_extracted.json",
        r"D:\UMER_ANF\Edubot_old_laptop\data\books\qanun_e_shahadat_sections_extracted_fixed.json"
    ]
    
    chroma_path = r"D:\UMER_ANF\Edubot_old_laptop\data\chroma"
    
    print("\n" + "=" * 60)
    print("LEGAL DOCUMENT INDEXING PIPELINE")
    print("=" * 60)
    
    indexer = LegalDocumentIndexer(json_files, chroma_path)
    indexer.load_json_files()
    chunks = indexer.chunk_documents(chunk_size=800, chunk_overlap=150)
    indexer.index_to_chroma(chunks)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
