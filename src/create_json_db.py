# import json
# import os
# import shutil
# import re
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# # ======================================================================
# # CONFIGURATION
# # ======================================================================

# JSON_FILES = [
#     r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\ppc_full_sections_left_superscript.json",
#     r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\cnsa_sections_extracted.json"
#     r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\amla_2010_final_structured.json"
#     r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\ANF_ACT_1997.json"
#     r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\punjab_police_rules_extracted.json"
#     r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\qanun_e_shahadat_sections_extracted_fixed.json"

# ]

# CHROMA_PATH = r"D:\Anti Narcotics Force\Edubot_old_laptop(working PPC Pydantic)\data\books\chroma"
# COLLECTION_NAME = "legal_sections"

# # ======================================================================
# # TEXT CLEANING FUNCTION
# # ======================================================================

# def clean_legal_text(text: str) -> str:
#     """Clean and normalize legal text for consistent embeddings."""
#     if not text:
#         return ""
#     text = re.sub(r'\s+', ' ', text)
#     text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
#     text = ' '.join(text.split())
#     text = re.sub(r'\s+([.,;:!?])', r'\1', text)
#     text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
#     return text.strip()

# # ======================================================================
# # LOAD & COMBINE JSON FILES
# # ======================================================================

# all_documents = []

# for path in JSON_FILES:
#     print(f"\n📂 Loading JSON file: {os.path.basename(path)}")
#     if not os.path.exists(path):
#         print(f"❌ File not found: {path}")
#         continue

#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     law_name = data.get("law_name", "Unknown Law")
#     sections = data.get("sections", [])
#     print(f"✅ Loaded {len(sections)} sections from {law_name}")

#     for s in sections:
#         section_num = s.get("section", "Unknown")
#         title = clean_legal_text(s.get("title", ""))
#         body = clean_legal_text(s.get("body", ""))
#         page = s.get("page", -1)
#         chapter = s.get("chapter", "")

#         if len(body) < 20:
#             continue

#         full_content = f"Section {section_num}: {title}\n\n{body}\n\nReference: {law_name}"
#         doc = Document(
#             page_content=full_content,
#             metadata={
#                 "section": str(section_num),
#                 "title": title,
#                 "body": body,
#                 "page": int(page) if isinstance(page, int) else -1,
#                 "chapter": chapter,
#                 "source": law_name,
#                 "law_name": law_name
#             }
#         )
#         all_documents.append(doc)

# print(f"\n✅ Total combined documents: {len(all_documents)}")

# # ======================================================================
# # SPLIT DOCUMENTS INTO CHUNKS
# # ======================================================================

# print("\n✂️ Splitting documents into chunks...")
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,
#     chunk_overlap=150,
#     add_start_index=True,
#     separators=["\n\n", "\n", ". ", " ", ""],
# )

# chunks = splitter.split_documents(all_documents)
# print(f"✅ Total chunks created: {len(chunks)}")

# # ======================================================================
# # RESET EXISTING CHROMA DATABASE
# # ======================================================================

# if os.path.exists(CHROMA_PATH):
#     shutil.rmtree(CHROMA_PATH)
#     print("🗑️ Removed old Chroma DB")

# # ======================================================================
# # INITIALIZE EMBEDDINGS
# # ======================================================================

# print("\n🤖 Initializing embedding model...")
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# # ======================================================================
# # SAVE TO CHROMA DATABASE
# # ======================================================================

# print("\n💾 Saving to Chroma database...")
# db = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory=CHROMA_PATH,
#     collection_name=COLLECTION_NAME
# )

# print(f"✅ Saved {len(chunks)} chunks to Chroma DB")
# print(f"📁 Database location: {CHROMA_PATH}")

# # ======================================================================
# # VERIFY STORED DATA
# # ======================================================================

# print("\n🔍 Verifying stored chunks...")

# # Reopen the same DB with the SAME embeddings
# db_verify = Chroma(
#     persist_directory=CHROMA_PATH,
#     embedding_function=embeddings,
#     collection_name=COLLECTION_NAME
# )

# count = db_verify._collection.count()
# print(f"✅ Verified stored chunks: {count}")

# # Show a few stored examples
# if count > 0:
#     docs = db_verify.get(limit=5)
#     print("\n📚 Sample stored metadata:")
#     for meta in docs['metadatas']:
#         print(f"• {meta.get('law_name')} → Section {meta.get('section')}")
# else:
#     print("⚠️ No chunks found — check file paths or embedding model setup.")

# print("\n🎯 Process complete.")


















import json
import os
import shutil
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ======================================================================
# CONFIGURATION
# ======================================================================

JSON_FILES = [
    r"D:\UMER_ANF\Edubot_old_laptop\data\books\ppc_full_sections_left_superscript.json",
    r"D:\UMER_ANF\Edubot_old_laptop\data\books\cnsa_sections_extracted.json",
    r"D:\UMER_ANF\Edubot_old_laptop\data\books\amla_2010_final_structured.json",
    r"D:\UMER_ANF\Edubot_old_laptop\data\books\ANF_ACT_1997.json",
    r"D:\UMER_ANF\Edubot_old_laptop\data\books\punjab_police_rules_extracted.json",
    r"D:\UMER_ANF\Edubot_old_laptop\data\books\qanun_e_shahadat_sections_extracted_fixed.json"
]

CHROMA_PATH = r"D:\UMER_ANF\Edubot_old_laptop\data\books\chroma"
COLLECTION_NAME = "legal_sections"

# ======================================================================
# TEXT CLEANING
# ======================================================================

def clean_legal_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = ' '.join(text.split())
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    return text.strip()

# ======================================================================
# FLEXIBLE SECTION EXTRACTOR
# ======================================================================

def extract_sections_from_json(data: dict):
    """Handle both 'sections' and nested 'volumes' structures."""
    sections = []

    if "sections" in data:
        sections.extend(data["sections"])

    if "volumes" in data and isinstance(data["volumes"], dict):
        for vol_name, vol_sections in data["volumes"].items():
            for sec in vol_sections:
                sec["volume"] = vol_name
                sections.append(sec)

    return sections

# ======================================================================
# MAIN COMBINATION LOGIC
# ======================================================================

all_documents = []

for path in JSON_FILES:
    print(f"\n📂 Loading JSON file: {os.path.basename(path)}")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    law_name = data.get("law_name", "Unknown Law")
    sections = extract_sections_from_json(data)
    print(f"✅ Loaded {len(sections)} sections from {law_name}")

    for s in sections:
        # Flexible section capture (handles Rule/Section/Article/1.1 etc.)
        raw_sec = str(s.get("section", "")).strip()
        sec_match = re.match(r'(?:Section|Article|Rule)?\s*([\dA-Za-z.\-() ]+)', raw_sec)
        section_num = sec_match.group(1).strip() if sec_match else raw_sec

        title = clean_legal_text(s.get("title", ""))
        body = clean_legal_text(s.get("body", ""))
        chapter = s.get("chapter", "") or s.get("part", "")
        volume = s.get("volume", "")
        page = s.get("page", -1)

        if len(body) < 20:  # lowered threshold
            continue

        # Construct a descriptive combined text
        section_label = f"Section {section_num}".strip()
        if "Rule" in raw_sec or "rule" in raw_sec.lower():
            section_label = f"Rule {section_num}"
        if "Article" in raw_sec or "article" in raw_sec.lower():
            section_label = f"Article {section_num}"

        full_content = f"{section_label}: {title}\n\n{body}\n\nReference: {law_name}"

        doc = Document(
            page_content=full_content,
            metadata={
                "section": section_num,
                "title": title,
                "body": body,
                "chapter": chapter,
                "volume": volume,
                "page": int(page) if isinstance(page, int) else -1,
                "law_name": law_name,
                "source": law_name
            }
        )
        all_documents.append(doc)

print(f"\n✅ Total combined documents: {len(all_documents)}")

# ======================================================================
# SPLIT INTO CHUNKS
# ======================================================================

print("\n✂️ Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(all_documents)
print(f"✅ Total chunks created: {len(chunks)}")

# ======================================================================
# RESET OLD CHROMA DB
# ======================================================================

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print("🗑️ Removed old Chroma DB")

# ======================================================================
# EMBEDDING MODEL
# ======================================================================

print("\n🤖 Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# ======================================================================
# SAVE TO CHROMA
# ======================================================================

print("\n💾 Saving to Chroma database...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name=COLLECTION_NAME
)

print(f"✅ Saved {len(chunks)} chunks to Chroma DB")
print(f"📁 Database location: {CHROMA_PATH}")

# ======================================================================
# VERIFY STORED DATA
# ======================================================================

print("\n🔍 Verifying stored chunks...")
db_verify = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

count = db_verify._collection.count()
print(f"✅ Verified stored chunks: {count}")

if count > 0:
    docs = db_verify.get(limit=5)
    print("\n📚 Sample stored metadata:")
    for meta in docs['metadatas']:
        print(f"• {meta.get('law_name')} → Section {meta.get('section')}")
else:
    print("⚠️ No chunks found — check file paths or embeddings.")

print("\n🎯 Process complete.")
