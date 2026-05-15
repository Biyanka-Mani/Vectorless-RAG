import os
import time
import json
from dotenv import load_dotenv
from pageindex import PageIndexClient
from openai import OpenAI

# ==========================================
# CONFIGURATION & CLIENTS
# ==========================================
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# High-quality FREE model on OpenRouter
LLM_MODEL = "openrouter/auto"

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ==========================================
# STEP 1: TREE SEARCH (Picking Node IDs)
# ==========================================

def tree_search(query, slim_tree):
    """LLM analyzes the tree and picks the most relevant node IDs."""
    print(f"\n🔍 Step 1: Searching tree for relevant sections...")
    prompt = f"""
    You are a document router. Analyze the document tree below and pick ONLY the IDs of the 
    MOST relevant sections (maximum 10) needed to answer the query.
    
    DOCUMENT TREE (IDs, Titles, Summaries):
    {json.dumps(slim_tree, indent=2)}
    
    QUERY: {query}
    
    TASK:
    Return a JSON array of unique node IDs. Do not repeat IDs.
    Example: ["0001", "0005"]
    """
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise router. You return ONLY a JSON list of unique IDs. No duplicates. No extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content
        
        # Robust extraction of the JSON list
        if "[" in content and "]" in content:
            ids_str = content[content.find("["):content.rfind("]")+1]
            ids = json.loads(ids_str)
            # De-duplicate and filter out non-string/malformed entries
            unique_ids = []
            for id in ids:
                if isinstance(id, str) and id not in unique_ids:
                    unique_ids.append(id)
            
            # Limit to top 10 most relevant to save tokens
            return unique_ids[:10]
        return []
    except Exception as e:
        print(f"❌ Tree Search Error: {e}")
        return []

# ==========================================
# HELPERS: TREE FLATTENING
# ==========================================

def flatten_tree(nodes, flat_map=None):
    """Converts a nested tree into a flat ID -> Node map for fast lookup."""
    if flat_map is None:
        flat_map = {}
    
    for node in nodes:
        node_id = node.get("node_id")
        if node_id:
            flat_map[node_id] = {
                "title": node.get("title"),
                "page": node.get("page_index"),
                "text": node.get("text"),
                "summary": node.get("summary") or node.get("prefix_summary") or "N/A"
            }
        
        if "nodes" in node and node["nodes"]:
            flatten_tree(node["nodes"], flat_map)
            
    return flat_map

# ==========================================
# STEP 2: RETRIEVE (Fetching from Flat Map)
# ==========================================

def retrieve_content(node_ids, flat_map):
    """Retrieves full text for selected nodes from the flattened tree map."""
    print(f"📥 Step 2: Retrieving full content for nodes: {node_ids}...")
    retrieved_context = []
    for node_id in node_ids:
        if node_id in flat_map:
            retrieved_context.append(flat_map[node_id])
        else:
            print(f"⚠️ Node {node_id} not found in document structure.")
    return retrieved_context

# ==========================================
# STEP 3: GENERATE (Grounded Answer)
# ==========================================

def grounded_generate(query, context):
    """Generates a grounded answer with citations and zero hallucination."""
    print(f"🧠 Step 3: Generating grounded answer...")
    if not context:
        return "❌ I couldn't find any relevant sections in the document to answer your question."

    context_str = ""
    for item in context:
        context_str += f"\n--- SECTION: {item['title']} (Page {item['page']}) ---\n{item['text']}\n"

    prompt = f"""
    You are a professional researcher. Use the PROVIDED CONTEXT below to answer the query.
    
    STRICT RULES:
    1. Base your answer ONLY on the provided context.
    2. If the answer is not in the context, say "I don't have enough information."
    3. Include page citations in your answer (e.g., [Page 12]).
    4. NO hallucinations. Do not use outside knowledge.
    
    CONTEXT FROM DOCUMENT:
    {context_str}
    
    QUERY: {query}
    """
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Generation Error: {str(e)}"

# ==========================================
# PIPELINE ORCHESTRATION
# ==========================================

def find_existing_document(filename):
    try:
        response = pi_client.list_documents()
        docs = response.get("documents", []) if isinstance(response, dict) else response
        target = os.path.splitext(filename)[0].lower()
        for doc in docs:
            if target in doc.get("name", "").lower():
                return doc.get("id")
    except: pass
    return None

def get_slim_tree(nodes):
    slim = []
    for node in nodes:
        item = {"id": node.get("node_id"), "title": node.get("title"), "summary": node.get("summary") or node.get("prefix_summary") or "N/A"}
        if "nodes" in node and node["nodes"]:
            item["children"] = get_slim_tree(node["nodes"])
        slim.append(item)
    return slim

def main():
    pdf_path = "./lessonplan.pdf"
    doc_id = find_existing_document(os.path.basename(pdf_path))
    
    if not doc_id:
        print("❌ Document not found. Please ensure it is uploaded.")
        return

    # Load the tree once
    print(f"📡 Loading document structure for {doc_id}...")
    tree_response = pi_client.get_tree(doc_id, node_summary=True)
    nodes = tree_response.get("result") or tree_response.get("nodes")
    
    # Create slim tree for Step 1 and flat map for Step 2
    slim_tree = get_slim_tree(nodes)
    flat_map = flatten_tree(nodes)

    print("\n✅ End-to-End Vectorless RAG Pipeline Ready!")
    print("💡 Type 'help' to see available commands.")
    
    while True:
        user_input = input("\n❓ Enter your question or command: ").strip()
        
        # 1. FLEXIBLE EXIT
        low_input = user_input.lower()
        if low_input in ["exit", "quit", "q"] or low_input.startswith("exit") or low_input.startswith("quit"): 
            print("👋 Goodbye!")
            break
        
        if not user_input: 
            continue

        # 2. HELP COMMAND
        if low_input == "help":
            print("\n🛠 AVAILABLE COMMANDS:")
            print("  - node <id> : Inspect a specific section (e.g., 'node 0005')")
            print("  - help      : Show this help message")
            print("  - quit/exit : Close the program")
            print("  - <anything else> : Ask a question to the RAG pipeline")
            continue

        # 3. NODE INSPECTION
        if low_input.startswith("node "):
            parts = user_input.split()
            if len(parts) > 1:
                target_id = parts[1]
                if target_id in flat_map:
                    node = flat_map[target_id]
                    print(f"\n🔍 INSPECTING NODE: {target_id}")
                    print("-" * 50)
                    print(f"Title:      {node['title']}")
                    print(f"Page:       {node['page']}")
                    print(f"\n[SUMMARY]:\n{node.get('summary', 'No summary available.')}")
                    print(f"\n[FULL CONTENT]:\n{node['text']}")
                    print("-" * 50)
                else:
                    print(f"❌ Node ID '{target_id}' not found.")
            else:
                print("❌ Please provide a node ID (e.g., 'node 0005')")
            continue # This continues the loop, asking for next input

        # 4. RAG SEARCH
        relevant_ids = tree_search(user_input, slim_tree)
        if not relevant_ids:
            print("❌ No relevant sections identified in the tree.")
            continue

        # 2. RETRIEVE
        full_context = retrieve_content(relevant_ids, flat_map)

        # 3. GENERATE
        answer = grounded_generate(user_input, full_context)
        print(f"\n💡 FINAL ANSWER:\n{answer}")

if __name__ == "__main__":
    main()
