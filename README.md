Ingests a markdown doc or folders of markdown docs and embeds them and indexes for searching. Can be prompted with a .txt of nlnl seperated queries for dataset generation. 

![dithered_image (4)](https://github.com/user-attachments/assets/665ef536-d4c7-45d3-b910-a37223df3fa3)

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

```
1. Contextual Relevance
   The QA pairs are directly tied to specific document chunks, ensuring relevance and accuracy.

   Document -> Chunks -> QA Pairs
   [A]        [B1][B2]   [C1][C2][C3]
               |  |      /  /  /
               |  |-----/--/--/
               |-------/--/
                       |

2. Diverse Query Formulations
   Query expansion creates varied question forms, improving model generalization.

   Original Query -> Expanded Queries
   [Q] ------------> [Q1][Q2][Q3]

3. Multi-turn Conversations
   Captures context-dependent interactions, enhancing conversational abilities.

   Turn 1    Turn 2    Turn 3
   Q1 -> A1 -> Q2 -> A2 -> Q3 -> A3
        |          |          |
        v          v          v
      Context    Context    Context

4. Grounded Responses
   Answers are based on document content, promoting factual accuracy.

   Document Content -> Retrieved Chunks -> Generated Answer
   [:::::::::::::]     [:::] [:::] [::]     [Answer Text]
                         |     |    |         ^  ^  ^
                         |     |    |---------|  |  |
                         |     |--------------|  |
                         |----------------------|

5. Metadata Rich
   Includes relevance scores and position information, useful for training retrieval models.

   Chunk: [Content] (Relevance: 0.95, Position: 156-234)

By finetuning models on this data, we can improve:
- Information retrieval capabilities
- Context-aware response generation
- Factual grounding in responses
- Handling of multi-turn conversations

This synthetic data bridges the gap between general language understanding and specific document comprehension, leading to more capable and reliable AI assistants.
```

```
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
  "max_documents_in_prompt": 3
}
```

# Configuration Options

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
