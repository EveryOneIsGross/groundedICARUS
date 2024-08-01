**note:** please excuse some very obvious frustrated workarounds, this was j meant to be an afternoons quick script. search will need tuning unless you are just golden running 🤙, but is close enough, numbers in relevence scores are vibes valued. chunking method needs a refactor it has some double handling, same with bm25 indexing being within the loop rather than at the end etc. 👽

---
groundedICARUS streamlines the creation of document-grounded datasets by processing markdown files, expanding queries, and generating synthetic QA pairs and conversations, offering flexible configuration and output options for enhanced lazy rag dataset generation.

![transparent_dithered_image](https://github.com/user-attachments/assets/f81a611f-6b2f-474d-a5e2-d7562564ee50)

Document Ingestion and Indexing:
1. The script can process either a single markdown document or a folder containing multiple markdown files.
2. Each document is split into chunks, and embeddings are generated for these chunks.
3. A hybrid search index is created, combining embedding-based similarity and BM25 text matching.

Query Processing:
1. Queries can be input interactively or loaded from a text file.
2. For dataset generation, a text file with queries separated by double newlines (nlnl) can be used.

Command-line Usage:

1. Basic usage with a single document:
   ```
   python script_name.py --input path/to/document.md
   ```

2. Processing a folder of markdown files:
   ```
   python script_name.py --input path/to/markdown/folder
   ```

3. Using a configuration file:
   ```
   python script_name.py --config path/to/config.json --input path/to/documents
   ```

4. Processing queries from a file:
   ```
   python script_name.py --input path/to/documents --query-file path/to/queries.txt
   ```

Query File Example (queries.txt):
```
What is the main topic of the document?

How does the author describe the problem?

What solutions are proposed in the text?

Can you summarize the key findings?

What evidence is presented to support the main argument?
```

Each query in this file is separated by a double newline (nlnl).

The script will process these queries sequentially, generating answers based on the ingested documents. This is particularly useful for creating large datasets of question-answer pairs grounded in specific documents.

Output:
1. The script generates JSONL files containing the query expansions and responses.
2. A markdown file is also created with a detailed log of the conversation, including system prompts, user queries, and AI responses.

This approach allows for efficient generation of document-grounded QA datasets, which can be valuable for fine-tuning language models or evaluating their performance on specific domains or document sets.


---

## FLOW

1. Hybrid Search:
   - Combines embedding similarity and BM25 for improved relevance

2. Dynamic Query Expansion:
   - Uses LLMs to generate context-aware, enhanced queries

3. Adaptive Chunking:
   - Creates variable-sized chunks based on content semantics

4. Multi-Modal Conversation Export:
   - Generates both JSONL for model training and Markdown for human review

5. Flexible API Integration:
   - Supports multiple LLM providers (OpenAI, Groq, Anthropic) and local models

6. Configurable Conversation Modes:
   - Switchable between zero-shot and multi-turn conversation logging

7. On-the-fly Embedding Generation:
   - Creates and caches embeddings for efficient reuse


```mermaid
graph TD
    A[Input] --> B[Document Ingestion]
    A --> C[Query Processing]
    
    B --> D[Chunking]
    D --> E[Embedding Generation]
    E --> F[Hybrid Search Index]
    
    C --> G[Query Expansion]
    G --> H[Enhanced Queries]
    
    F --> I[Retrieval]
    H --> I
    
    I --> J[Context Formation]
    J --> K[Answer Generation]
    
    K --> L[Multi-Modal Output]
    L --> M[JSONL for Training]
    L --> N[Markdown for Review]
    
    O[Configuration] --> B
    O --> C
    O --> D
    O --> E
    O --> F
    O --> G
    O --> I
    O --> J
    O --> K
    O --> L
    
    P[API Integration] --> G
    P --> K
    
    Q[Embedding Cache] --> E
    E --> Q
    
    R[Conversation History] --> J
    K --> R

    subgraph "Data Flow"
        A
        B
        C
        D
        E
        F
        G
        H
        I
        J
        K
        L
        M
        N
    end

    subgraph "Configuration & Integration"
        O
        P
        Q
        R
    end
```

# Configuration Options

```config.json
{
  "model_name": "orca-mini-3b-gguf2-q4_0.gguf",
  "embedding_resolution": 64,
  "chunk_size": 256,
  "chunk_overlap": 16,
  "history_window": 3,
  "system_prompt_file": "",
  "query_expansion_system_prompt_file": "",
  "query_expansion_prompt_file": "",
  "max_doc_length": 256,
  "top_k": 10,
  "relevance_threshold": 0.5,
  "include_full_content": false,
  "truncate_start": true,
  "query_expansion_output": "query_expansion.jsonl",
  "response_output": "response.jsonl",
  "api_type": "openai",
  "api_base_url": "http://localhost:11434/v1",
  "api_key": "ollama",
  "query_expansion_model": "llama3.1:latest",
  "answer_generation_model": "llama3.1:latest",
  "use_conversation_history": false,
  "include_position": false,
  "max_documents_in_prompt": 3,
  "zero_shot": true,
}
```

## Model Settings

- `model_name`: The name of the local model to use (e.g., "orca-mini-3b-gguf2-q4_0.gguf"). This is used for the GPT4All model.
- `query_expansion_model`: The model to use for query expansion. Choose based on your API and available models.
- `answer_generation_model`: The model to use for answer generation. Choose based on your API and available models.

## API Settings

- `api_type`: The type of API to use. Options: 'openai', 'anthropic', 'groq', or 'ollama'.
- `api_base_url`: The base URL for the API. For local Ollama, use 'http://localhost:11434/v1'. For OpenAI, use 'https://api.openai.com/v1'.
- `api_key`: The API key. Use 'ollama' for local Ollama, or your actual API key for other services.

## Embedding Settings

- `embedding_resolution`: The dimensionality of the embedding vectors. Higher values may provide more accurate but slower embeddings. (Default: 64)

## Chunking Settings

- `chunk_size`: The size of text chunks for processing. Larger chunks provide more context but may be slower to process. (Default: 256)
- `chunk_overlap`: The number of tokens to overlap between chunks. Higher overlap may improve coherence but increases processing time. (Default: 16)

## Conversation Settings

- `history_window`: The number of previous conversation turns to include for context. Larger windows provide more context but may slow down processing. (Default: 3)
- `use_conversation_history`: Whether to use conversation history for context in responses. (Options: true/false)
- `zero_shot`: Whether it saves as a 0-shot q&a pair to jsonl or as a single multi-turn conversation. (Options: true/false)

## Document Settings

- `max_doc_length`: The maximum length of document content to include in prompts. Longer lengths provide more context but may exceed token limits. (Default: 256)
- `include_full_content`: Whether to include the full document content in prompts. Enabling may provide more context but significantly increases token usage. (Options: true/false)
- `truncate_start`: Whether to truncate content from the start (true) or end (false) when reducing to max_doc_length. (Options: true/false)
- `max_documents_in_prompt`: The maximum number of documents to include in the prompt. More documents provide broader context but increase token usage. (Default: 3)

## Retrieval Settings

- `top_k`: The number of top-ranked chunks to retrieve. Higher values may improve recall but increase processing time and token usage. (Default: 10)
- `relevance_threshold`: The minimum relevance score for a chunk to be included. Higher values increase precision but may reduce recall. (Default: 0.5)

## Output Settings

- `include_position`: Whether to include position information (start/end) in the output for each chunk. (Options: true/false)
- `query_expansion_output`: The filename for saving query expansion data. (Default: "query_expansion.jsonl")
- `response_output`: The filename for saving response data. (Default: "response.jsonl")

## Prompt Settings

- `system_prompt_file`: Path to the file containing the system prompt. If empty, a default prompt will be used.
- `query_expansion_system_prompt_file`: Path to the file containing the system prompt for query expansion. If empty, a default prompt will be used.
- `query_expansion_prompt_file`: Path to the file containing the query expansion prompt. If empty, a default prompt will be used.

To use groundedICARUS, prepare three separate markdown (.md) files for its key prompts:

1. Query Expansion System Prompt: A general instruction for expanding queries.
2. Query Expansion Prompt: Include placeholders {source} and {documents} for context. This guides the query enhancement process.
3. Q&A System Prompt: Directs the final answer generation step.

note: JSONL excludes search enhanced query and saves it seperately, without distorting the initial user query. 

  ```
  {
  "id": 0,
  "conversations": [
    {
      "from": "system",
      "value": "System prompt content"
    },
    {
      "from": "human",
      "value": "User query with formatted retrival docs"
    },
    {
      "from": "gpt",
      "value": "AI-generated answer",
      "weight": 1
    }
  ],
  "docs": [
    {
      "content": "Relevant chunk of text from the document",
      "source": "filename.md",
      "start": 0,
      "end": 500,
      "relevance": 0.95
    }
  ]
}
```

![dithered_image (4)](https://github.com/user-attachments/assets/665ef536-d4c7-45d3-b910-a37223df3fa3)
