# Page Index: Vectorless RAG Pipeline

This project implements a high-precision, **Vectorless RAG (Retrieval-Augmented Generation)** pipeline using the Page Index API and OpenRouter (for LLM access). Unlike traditional RAG, this system understands the **hierarchical structure** of documents, leading to higher accuracy and lower token costs.

## 🚀 The 3-Step RAG Pipeline

The `main.py` script follows a structured "Search-Retrieve-Generate" flow:

### 1. Tree Search (Reasoning Phase)
- **Method:** `tree_search(query, slim_tree)`
- **Relevance:** Instead of searching through millions of text chunks, the LLM first looks at a "Slim Tree" (Titles + Summaries only). 
- **Goal:** The LLM acts as a "Router" and identifies the exact `node_id`s that contain the information needed to answer the user's question.

### 2. Retrieve (Data Phase)
- **Method:** `retrieve_content(node_ids, flat_map)`
- **Relevance:** Once the IDs are selected, we pull the **actual full text** from a local `flat_map`.
- **Optimization:** We use `flatten_tree()` to turn the nested API response into a fast lookup table, making this step nearly instantaneous and avoiding redundant API calls.

### 3. Generate (Synthesis Phase)
- **Method:** `grounded_generate(query, context)`
- **Relevance:** The LLM receives the full text of only the relevant sections.
- **Strict Rules:** It is instructed to answer **only** using the provided context, include **[Page X] citations**, and strictly avoid hallucinations.

---

## 🛠 Internal Method Glossary

### Page Index API Methods
- `list_documents()`: Used to scan the account and find existing PDFs. This prevents redundant uploads and preserves  page limits.
- `get_tree(doc_id, node_summary=True)`: Retrieves the entire document hierarchy. It includes titles, page numbers, full text, and semantic summaries for every section.
- `get_document(doc_id)`: Checks the indexing status to ensure the document is ready for queries.

### Pipeline Helper Methods
- `flatten_tree(nodes)`: A recursive function that transforms the nested tree into a flat `ID -> Content` dictionary. This is why we don't need a `get_node` API call; the text is already locally available for $O(1)$ lookup.
- `get_slim_tree(nodes)`: Removes the heavy `text` fields from the tree, leaving only IDs, Titles, and Summaries. This is sent to the LLM in Step 1 to save **70-90% in token costs**.
- `find_existing_document(filename)`: A smart matching helper that finds  file on the server by name.

---

## 🌐 OpenRouter Integration
This project uses **OpenRouter** to access high-quality LLMs for free.

---

## 📋 Setup & Usage

1. **Environment Variables**: Create a `.env` file:
   ```env
   PAGEINDEX_API_KEY=your_pageindex_key
   OPENROUTER_API_KEY=your_openrouter_key
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   ```bash
   python main.py
   ```

