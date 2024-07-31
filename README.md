![dithered_image (4)](https://github.com/user-attachments/assets/665ef536-d4c7-45d3-b910-a37223df3fa3)


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
