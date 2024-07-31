import os
import pickle
import json
from typing import List, Dict
from gpt4all import GPT4All
from nomic import embed
import numpy as np
from openai import OpenAI
from colorama import init, Fore, Style
import argparse
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import tiktoken
import nltk

# nltk.download('punkt', quiet=True)

# Anthropic client
from anthropic import Anthropic

# Groq client
from groq import Groq

# import .env
from dotenv import load_dotenv
load_dotenv()

OPENAI_CLIENT = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
GROQ_CLIENT = Groq(api_key=os.getenv('GROQ_API_KEY'))
ANTHROPIC_CLIENT = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
API_CLIENT = GROQ_CLIENT

init(autoreset=True)  # Initialize colorama

class ConversationalDocQAAgent:
    def __init__(self, config: dict):
        self.config = config
        self.client = self.get_api_client()
        self.model_name = config.get('model_name', 'orca-mini-3b-gguf2-q4_0.gguf')
        self.query_expansion_model = config.get('query_expansion_model', 'gemma2:latest')
        self.answer_generation_model = config.get('answer_generation_model', 'interstellarninja/hermes-2-pro-llama-3-8b:latest')
        self.llm = GPT4All(self.model_name)
        self.max_doc_length = config.get('max_doc_length', 1028)
        self.include_full_content = config.get('include_full_content', False)
        self.truncate_start = config.get('truncate_start', True)
        self.all_tokenized_chunks = []
        self.documents: List[Dict] = []
        self.max_documents_in_prompt = config.get('max_documents_in_prompt', 3)  # Default to 3 documents
        self.embeddings: List[np.ndarray] = []
        self.document_contents: Dict[str, str] = {}
        self.chunk_size = config.get('chunk_size', 1028)
        self.chunk_overlap = config.get('chunk_overlap', 128)
        self.embedding_resolution = config.get('embedding_resolution', 64)
        self.include_position = config.get('include_position', True)
        self.top_k = config.get('top_k', 10)
        self.bm25 = None
        self.tokenized_chunks = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.relevance_threshold = config.get('relevance_threshold', 0.5)
        self.conversation_history: List[Dict] = []
        self.history_window = config.get('history_window', 3)
        self.system_prompt_template = self.load_prompt(config.get('system_prompt_file', 'SYSPROMPTS\\system-prompt-hermesREASONRAG.md'), "system")
        self.query_expansion_prompt = self.load_prompt(config.get('query_expansion_prompt_file', 'SYSPROMPTS\\system-prompt-queryexpansion_graph_banana.md'), "query expansion")
        self.query_expansion_system_prompt = self.load_prompt(config.get('query_expansion_system_prompt_file', 'SYSPROMPTS\\system-prompt-NUFORMSqueryexpansion.md'), "query expansion system")
        self.query_expansion_data = []
        self.response_data = []
        self.conversation_id = 0
        self.export_data = []
        self.setup_logging()
        self.markdown_conversation = []

    def setup_logging(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, "qa_agent.log")
        
        self.logger = logging.getLogger("QAAgent")
        self.logger.setLevel(logging.DEBUG)

        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_api_client(self):
        api_type = self.config.get('api_type', 'openai').lower()
        if api_type == 'openai':
            return OpenAI(base_url=self.config.get('api_base_url', 'http://localhost:11434/v1'), 
                        api_key=self.config.get('api_key', 'ollama'))
        elif api_type == 'groq':
            return Groq(api_key=self.config.get('api_key', os.getenv('GROQ_API_KEY')))
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    def load_prompt(self, filename: str, prompt_type: str) -> str:
        try:
            print(f"{Fore.YELLOW}Loading {prompt_type} prompt from: {filename}")
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"{Fore.GREEN}{prompt_type.capitalize()} prompt loaded successfully")
            return content
        except FileNotFoundError:
            print(f"{Fore.RED}Error: {prompt_type.capitalize()} prompt file '{filename}' not found.")
            print(f"{Fore.YELLOW}Using default {prompt_type} prompt.")
            return self.get_default_prompt(prompt_type)



    def get_default_prompt(self, prompt_type: str) -> str:
        if prompt_type == "system":
            return """
You're an expert research assistant. Answer questions about the following documents:

Process:

List numbered, relevant quotes from the document. If none, state "No relevant quotes".
Answer based on quotes, referencing them with [#].

Format:
Quotes:
[1] "[Quote]"
[2] "[Quote]"
Answer: [Response with [#] references]
If unanswerable, state so.
"""
        elif prompt_type == "query":
            return """Based on the following relevant information from the document, please answer the question:

<documents>
Relevant Docs:
{formatted_chunks}
</documents>

{conversation_history}

Your reasoned responses should be concise, well-formatted, and always include proper citations.

Question: {user_question}

Answer: [Your answer goes here]"""
        elif prompt_type == "query expansion":
            return """
Generate a comprehensive query for information retrieval based on:

1. <source>{source}</source>
2. <documents>{document_content}</documents>
3. User's Question: 
<user_question>
{conversation_history}

{user_question}
</user_question>

Process:
1. Analyze question, history, content, and source.
2. Identify key concepts, entities, and related topics.
3. Create a verbose, keyword-rich query incorporating all elements.

Output the enhanced query directly, without preamble:

[Verbose, keyword-rich query]
"""
        elif prompt_type == "query expansion system":
            return """
You are an AI assistant specialized in expanding user queries for improved information retrieval. 
Your task is to analyze the given question, context, and document information to generate a comprehensive, 
keyword-rich query that will maximize the retrieval of relevant information.
"""
        else:
            return "Default prompt not available for this type."


    def process_folder(self, folder_path: str):
        processed_files = 0
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.md'):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, "r", encoding='utf-8') as f:
                            content = f.read()
                        self.add_document(content, filename, file_path)
                        processed_files += 1
                        print(f"{Fore.GREEN}Processed: {file_path}")
                    except Exception as e:
                        print(f"{Fore.RED}Error processing {file_path}: {str(e)}")
        
        if processed_files > 0:
            self.recreate_bm25_index()
            print(f"{Fore.GREEN}Successfully processed {processed_files} .md files in: {folder_path}")
        else:
            print(f"{Fore.YELLOW}No .md files found in: {folder_path}")

    def add_document(self, content: str, filename: str, file_path: str):
        self.logger.info(f"Processing document: {filename}")
        self.logger.info(f"File path: {file_path}")
        
        # Generate the cache file path
        cache_file = os.path.splitext(file_path)[0] + '.pkl'
        self.logger.info(f"Checking for cache file: {cache_file}")
        
        if os.path.exists(cache_file):
            self.logger.info(f"Cache file found: {cache_file}")
            print(f"{Fore.GREEN}Loading cached data for {filename}")
            self.load_cached_data(cache_file, filename)
            return

        self.logger.info(f"No cache file found. Processing document: {filename}")
        print(f"{Fore.YELLOW}Processing document: {filename}")
        
        try:
            chunks = self.create_chunks(content, filename)
            
            # Log before starting embedding process
            self.logger.info(f"Starting embedding process for {filename}")
            print(f"{Fore.YELLOW}Creating embeddings for: {filename}")
            
            embeddings = self.create_embeddings(chunks)
            tokenized_chunks = [self.tokenize(chunk['content']) for chunk in chunks]
            
            self.documents.extend(chunks)
            self.embeddings.extend(embeddings)
            self.all_tokenized_chunks.extend(tokenized_chunks)
            
            self.document_contents[filename] = content  # Use filename as key

            self.logger.info(f"Caching data for {filename}")
            print(f"{Fore.YELLOW}Caching data for {filename}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'tokenized_chunks': tokenized_chunks,
                    'full_content': content
                }, f)
            self.logger.info(f"Cache file created: {cache_file}")
            print(f"{Fore.GREEN}Cached data for {filename}")

            print(f"{Fore.GREEN}Document processing complete: {filename}")
            print(f"Total chunks: {len(chunks)}")
            print(f"Total tokens: {sum(len(tc) for tc in tokenized_chunks)}")

            # Recreate BM25 index after adding the document
            self.recreate_bm25_index()

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def load_cached_data(self, cache_file: str, filename: str):
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            self.documents.extend(cached_data['chunks'])
            self.embeddings.extend(cached_data['embeddings'])
            self.all_tokenized_chunks.extend(cached_data['tokenized_chunks'])
            self.document_contents[filename] = cached_data.get('full_content', '')  # Use filename as key
        print(f"{Fore.GREEN}Loaded cached data for {filename}")


    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def create_chunks(self, content: str, source: str) -> List[Dict]:
        sentences = nltk.sent_tokenize(content)
        chunks = []
        current_chunk = ""
        current_chunk_start = 0
        
        def finalize_chunk(chunk, start, end):
            return {
                "content": self.clean_text(chunk),
                "source": source,
                "start": start,
                "end": end
            }

        def token_len(text):
            return len(self.tokenize(text))

        for i, sentence in enumerate(sentences):
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if token_len(potential_chunk) > self.chunk_size:
                if not current_chunk:
                    chunks.append(finalize_chunk(sentence, current_chunk_start, current_chunk_start + len(sentence)))
                    current_chunk_start = current_chunk_start + len(sentence)
                    continue
                
                if token_len(current_chunk) >= self.chunk_size * 0.8:
                    chunks.append(finalize_chunk(current_chunk, current_chunk_start, current_chunk_start + len(current_chunk)))
                    current_chunk = sentence
                    current_chunk_start = content.index(sentence)
                else:
                    next_sentence = sentences[i+1] if i+1 < len(sentences) else ""
                    if token_len(potential_chunk + next_sentence) <= self.chunk_size * 1.2:
                        current_chunk = potential_chunk
                    else:
                        chunks.append(finalize_chunk(current_chunk, current_chunk_start, current_chunk_start + len(current_chunk)))
                        current_chunk = sentence
                        current_chunk_start = content.index(sentence)
            else:
                current_chunk = potential_chunk

        if current_chunk:
            chunks.append(finalize_chunk(current_chunk, current_chunk_start, current_chunk_start + len(current_chunk)))

        return chunks

    def chunk_embedding(self, text: str, resolution: int = None) -> List[np.ndarray]:
        """
        Create embeddings for chunks of text at a specified resolution using tiktoken.
        
        Args:
        text (str): The input text to be chunked and embedded.
        resolution (int): The desired chunk size for embedding in tokens. 
                        If None, uses the value from config.
        
        Returns:
        List[np.ndarray]: A list of embeddings for each chunk.
        """
        if resolution is None:
            resolution = self.embedding_resolution
        
        tokens = self.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), resolution):
            chunk = self.tokenizer.decode(tokens[i:i+resolution])
            chunks.append(chunk)
        
        # Create embeddings for each chunk
        embeddings = embed.text(chunks, model="nomic-embed-text-v1.5", inference_mode="local")['embeddings']
        
        return embeddings

    def create_embeddings(self, chunks: List[Dict]) -> List[np.ndarray]:
        embeddings = []
        for chunk in tqdm(chunks, desc="Creating embeddings", unit="chunk"):
            # Use the updated chunk_embedding method without specifying resolution
            chunk_embeddings = self.chunk_embedding(chunk['content'])
            # If multiple embeddings are returned for a chunk, we average them
            embedding = np.mean(chunk_embeddings, axis=0) if len(chunk_embeddings) > 1 else chunk_embeddings[0]
            embeddings.append(embedding)
        return embeddings

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\([^()]*?_image_\d+\.png\)', '', text)
        # Replace multiple spaces with a single space, but preserve newlines
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with a maximum of two newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in {'.', '!', '?'}:
            text += '.'
        return text.strip()  # Only strip leading/trailing whitespace

    def toggle_position_inclusion(self):
        self.config['include_position'] = not self.config.get('include_position', True)
        state = "included" if self.config['include_position'] else "excluded"
        print(f"{Fore.GREEN}Position information will now be {state} in the prompts.")

    def populate_system_prompt(self, template: str) -> str:
        document_summaries = []
        for source, content in list(self.document_contents.items())[:self.max_documents_in_prompt]:
            if self.include_full_content:
                document_summaries.append(f"Source: {source}\nContent: {content}")
            else:
                truncated_content = self.truncate_document(content, max_length=500)
                document_summaries.append(f"Source: {source}\nContent: {truncated_content}")
        
        all_documents_summary = "\n\n".join(document_summaries)
        
        replacements = {
            'source': f"Multiple documents (showing {len(document_summaries)} of {len(self.document_contents)})",
            'document_content': all_documents_summary
        }
        
        populated_prompt = template
        for key, value in replacements.items():
            placeholder = '{' + key + '}'
            if placeholder in populated_prompt:
                populated_prompt = populated_prompt.replace(placeholder, str(value))
        
        return populated_prompt
    
    def prepare_conversation_history(self) -> str:
            if not self.config.get("use_conversation_history", False):
                return ""
            
            history = "\n".join([
                f"Q: {turn['content']}\nA: {turn['answer']}"
                for turn in self.conversation_history[-self.history_window:]
            ])
            
            return f"Conversation History:\n{history}" if history else ""

    def generate_query(self, user_question: str) -> str:
        self.logger.info(f"Generating expanded query for: {user_question}")
        
        conversation_history_str = self.prepare_conversation_history()
        
        all_documents_summary = "\n\n".join([
            f"Source: {source}\nContent: {self.truncate_document(content, max_length=500)}"
            for source, content in list(self.document_contents.items())[:self.max_documents_in_prompt]
        ])
        
        query_prompt = self.query_expansion_prompt.format(
            user_question=user_question,
            conversation_history=conversation_history_str,
            source=f"Multiple documents (showing {min(self.max_documents_in_prompt, len(self.document_contents))} of {len(self.document_contents)})",
            document_content=all_documents_summary
        )
        
        populated_system_prompt = self.populate_system_prompt(self.query_expansion_system_prompt)
        
        print(f"\n{Fore.YELLOW}Query Expansion System Prompt:")
        print(f"{Fore.WHITE}{populated_system_prompt}")
        
        print(f"\n{Fore.YELLOW}Query Expansion User Prompt:")
        print(f"{Fore.WHITE}{query_prompt}")
        
        print(f"{Fore.YELLOW}Generating expanded query...")
        response = self.client.chat.completions.create(
            model=self.query_expansion_model,
            messages=[
                {"role": "system", "content": populated_system_prompt},
                {"role": "user", "content": query_prompt}
            ]
        )
        
        generated_query = response.choices[0].message.content.strip()
        print(f"{Fore.GREEN}Query generated successfully")
        print(f"{Fore.CYAN}Generated Query: {generated_query}")
        
        return generated_query

    def add_to_query_expansion_export(self, original_question: str, expanded_query: str, query_prompt: str, system_prompt: str, relevant_chunks: List[Dict]):
        export_entry = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": query_prompt},
                {"from": "gpt", "value": expanded_query, "weight": 1}
            ],
            "docs": [
                {
                    "content": chunk['content'],
                    "source": chunk['source'],
                    "start": chunk['start'],
                    "end": chunk['end'],
                    "relevance": chunk['relevance']
                } for chunk in relevant_chunks
            ],
            "id": self.conversation_id,
            "populated_system_prompt": system_prompt,
            "populated_query_prompt": query_prompt
        }
        self.query_expansion_data.append(export_entry)
        self.conversation_id += 1

    def recreate_bm25_index(self):
        print(f"{Fore.YELLOW}Recreating BM25 index for all documents...")
        decoded_chunks = [self.tokenizer.decode(tc) for tc in self.all_tokenized_chunks]
        self.bm25 = BM25Okapi(decoded_chunks)
        print(f"{Fore.GREEN}BM25 index recreated successfully")

    def search_documents(self, query: str) -> List[Dict]:
        query_embedding = embed.text([query], model="nomic-embed-text-v1.5", inference_mode="local")['embeddings'][0]
        embedding_similarities = [np.dot(query_embedding, doc_embedding) for doc_embedding in self.embeddings]
        
        tokenized_query = self.tokenize(query)
        tokenized_query_str = self.tokenizer.decode(tokenized_query)
        
        if self.bm25 is None:
            self.logger.warning("BM25 index is None. Recreating index.")
            self.recreate_bm25_index()
        
        if self.bm25 is None:
            self.logger.error("Failed to create BM25 index. Falling back to embedding similarity only.")
            bm25_scores = np.zeros_like(embedding_similarities)
        else:
            bm25_scores = self.bm25.get_scores(tokenized_query_str.split())
        
        embedding_range = np.max(embedding_similarities) - np.min(embedding_similarities)
        if embedding_range == 0:
            embedding_scores = np.ones_like(embedding_similarities)
        else:
            embedding_scores = (embedding_similarities - np.min(embedding_similarities)) / embedding_range

        bm25_range = np.max(bm25_scores) - np.min(bm25_scores)
        if bm25_range == 0:
            bm25_scores_normalized = np.ones_like(bm25_scores)
        else:
            bm25_scores_normalized = (bm25_scores - np.min(bm25_scores)) / bm25_range
        
        combined_scores = 0.5 * embedding_scores + 0.5 * bm25_scores_normalized
        
        # Sort all indices by score in descending order
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        retrieved_chunks = []
        for i in sorted_indices:
            relevance = round(combined_scores[i], 2)
            
            if relevance < self.relevance_threshold:
                continue
            
            original_chunk = self.documents[i]
            source = os.path.basename(original_chunk['source'])

            
            context_start = self.find_boundary(original_chunk['start'], False, source)
            context_end = self.find_boundary(original_chunk['end'], True, source)
            
            while context_end - context_start < self.chunk_size and (context_start > 0 or context_end < len(self.document_contents[source])):
                if context_start > 0:
                    context_start = self.find_boundary(context_start - 1, False, source)
                if context_end < len(self.document_contents[source]):
                    context_end = self.find_boundary(context_end + 1, True, source)
            
            extended_content = self.document_contents[source][context_start:context_end]
            
            retrieved_chunks.append({
                "content": self.clean_text(extended_content),
                "source": source,
                "start": context_start,
                "end": context_end,
                "relevance": relevance
            })
            
            # Break if we've reached the top_k limit
            if len(retrieved_chunks) >= self.top_k:
                break
        
        merged_chunks = self.merge_overlapping_chunks(retrieved_chunks)
        
        # Ensure we return at most top_k chunks
        return merged_chunks[:self.top_k]

    def find_boundary(self, index: int, looking_forward: bool, source: str) -> int:
        """Find the nearest sentence or paragraph boundary."""
        content = self.document_contents[source]  # source is now the filename
        if looking_forward:
            while index < len(content) and content[index] not in {'.', '!', '?', '\n'}:
                index += 1
            index += 1
        else:
            while index > 0 and content[index - 1] not in {'.', '!', '?', '\n'}:
                index -= 1
        return index

    def merge_overlapping_chunks(self, chunks: List[Dict]) -> List[Dict]:
        # Sort chunks by relevance (descending) and then by source and start position
        sorted_chunks = sorted(chunks, key=lambda x: (-x['relevance'], x['source'], x['start']))
        merged = []
        
        for chunk in sorted_chunks:
            if not merged or chunk['source'] != merged[-1]['source'] or chunk['start'] > merged[-1]['end']:
                merged.append(chunk)
            else:
                merged[-1]['end'] = max(merged[-1]['end'], chunk['end'])
                merged[-1]['content'] = self.document_contents[chunk['source']][merged[-1]['start']:merged[-1]['end']]
                merged[-1]['relevance'] = max(merged[-1]['relevance'], chunk['relevance'])
        
        return merged
    
    def format_chunks(self, chunks: List[Dict]) -> str:
        formatted_chunks = []
        include_position = self.config.get("include_position", True)
        for i, chunk in enumerate(chunks, start=0):
            relevance = chunk['relevance']
            score_str = f"{relevance:.2f}" if not np.isnan(relevance) else "N/A"
            
            source_filename = chunk['source']  # Already a filename, no need to use os.path.basename
            # remove the extension
            source_filename = os.path.splitext(source_filename)[0]

            cleaned_content = self.clean_text(chunk['content'])
            
            chunk_info = f"[{i}] \"{cleaned_content}\""
            
            if include_position:
                chunk_info += f" (Source: {source_filename}, Position: {chunk['start']}-{chunk['end']}, Score: {score_str})"
            else:
                chunk_info += f" (Source: {source_filename}, Score: {score_str})"
            
            formatted_chunks.append(chunk_info)
        
        return "\n\n".join(formatted_chunks)

    def truncate_document(self, text: str, max_length: int = None, truncate_start: bool = None) -> str:
        if truncate_start is None:
            truncate_start = self.truncate_start
        if max_length is None:
            max_length = self.max_doc_length
        if len(text) <= max_length:
            return text
        
        if truncate_start:
            truncated_text = text[:max_length]
            return f"{truncated_text}\n\n  ... "
        else:
            half_length = max_length // 2
            first_part = text[:half_length]
            last_part = text[-half_length:]
            return f"{first_part}\n\n  cont...  \n\n{last_part}"

    def answer_question(self, user_question: str) -> str:
        self.logger.info(f"Answering question: {user_question}")
        try:
            generated_query = self.generate_query(user_question)
            relevant_chunks = self.search_documents(generated_query)
            
            if not relevant_chunks:
                return "I was unable to find relevant information in the documents. Falling back on base knowledge to conservatively answer the question."
            
            formatted_chunks = self.format_chunks(relevant_chunks)
            populated_system_prompt = self.populate_system_prompt(self.system_prompt_template)
            
            conversation_history_str = self.prepare_conversation_history()
            
            query_prompt_template = self.get_default_prompt("query")
            
            format_dict = {
                "formatted_chunks": formatted_chunks,
                "conversation_history": conversation_history_str,
                "user_question": user_question
            }
            
            query_prompt = query_prompt_template.format_map(type('SafeDict', (dict,), {'__missing__': lambda self, key: '{' + key + '}'})(**format_dict))

            self.add_to_query_expansion_export(user_question, generated_query, query_prompt, populated_system_prompt, relevant_chunks)

            print(f"\n{Fore.YELLOW}Answer Generation System Prompt:")
            print(f"{Fore.WHITE}{populated_system_prompt}")

            print(f"\n{Fore.YELLOW}Answer Generation User Prompt:")
            print(f"{Fore.WHITE}{query_prompt}")

            response = self.client.chat.completions.create(
                model=self.answer_generation_model,
                messages=[
                    {"role": "system", "content": populated_system_prompt},
                    {"role": "user", "content": query_prompt}
                ]
            )
            
            answer = response.choices[0].message.content

            cleaned_answer = re.sub(r'\n{3,}', '\n\n', answer.strip())
            
            if self.config.get("use_conversation_history", False):
                compact_answer = cleaned_answer.replace('\n', ' ')
                self.conversation_history.append({"content": user_question.replace('\n', ' '), "answer": compact_answer})
                self.conversation_history = self.conversation_history[-self.history_window:]
            
            self.add_to_response_export(user_question, cleaned_answer, relevant_chunks, populated_system_prompt, query_prompt, generated_query)
            
            self.markdown_conversation.append({
                "role": "human",
                "content": user_question
            })
            self.markdown_conversation.append({
                "role": "assistant",
                "content": cleaned_answer
            })
            
            return cleaned_answer
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}", exc_info=True)
            return f"An error occurred while processing your question: {str(e)}"

    def add_to_response_export(self, question: str, answer: str, relevant_chunks: List[Dict], system_prompt: str, query_prompt: str, generated_query: str):
        export_entry = {
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": query_prompt},
                {"from": "gpt", "value": answer, "weight": 1}
            ],
            "docs": [
                {
                    "content": chunk['content'],
                    "source": chunk['source'],
                    "start": chunk['start'],
                    "end": chunk['end'],
                    "relevance": chunk['relevance']
                } for chunk in relevant_chunks
            ],
            "id": self.conversation_id,
            "populated_system_prompt": system_prompt,
            "populated_query_prompt": query_prompt
        }
        self.response_data.append(export_entry)
        self.conversation_id += 1

    def export_to_jsonl(self, query_expansion_filename: str, response_filename: str):
        with open(query_expansion_filename, 'w', encoding='utf-8') as f:
            for entry in self.query_expansion_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"{Fore.GREEN}Exported query expansion data to {query_expansion_filename}")

        with open(response_filename, 'w', encoding='utf-8') as f:
            for entry in self.response_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"{Fore.GREEN}Exported response data to {response_filename}")

    def export_markdown_conversation(self, filename: str):
        self.logger.info(f"Exporting markdown conversation to {filename}")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Conversation with Document QA Agent\n\n")
                f.write(f"Documents processed:\n")
                for source in self.document_contents.keys():
                    f.write(f"- {source}\n")
                f.write(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if self.query_expansion_data:
                    f.write("## System Prompt\n\n")
                    f.write(f"```\n{self.query_expansion_data[0]['populated_system_prompt']}\n```\n\n")
                
                for i, (query, response) in enumerate(zip(self.query_expansion_data, self.response_data)):
                    f.write(f"## Conversation Turn {i+1}\n\n")
                    
                    f.write("### User Question\n\n")
                    f.write(f"{query['conversations'][1]['value']}\n\n")
                    
                    f.write("### Query Expansion\n\n")
                    f.write(f"Expanded Query: {query['conversations'][2]['value']}\n\n")
                    
                    f.write("### Retrieved Chunks\n\n")
                    for chunk in query['docs']:
                        f.write(f"- {chunk['content']} (Source: {chunk['source']}, Relevance: {chunk['relevance']})\n")
                    f.write("\n")
                    
                    f.write("### LLM Prompt\n\n")
                    f.write(f"```\n{response['populated_query_prompt']}\n```\n\n")
                    
                    f.write("### Assistant Response\n\n")
                    f.write(f"{response['conversations'][2]['value']}\n\n")
                    
                    f.write("---\n\n")
                
            self.logger.info(f"Markdown conversation exported successfully")
        except Exception as e:
            self.logger.error(f"Error exporting markdown conversation: {str(e)}", exc_info=True)


    def process_queries_from_file(self, query_file: str):
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                queries = f.read().split('\n\n')
            
            for i, query in enumerate(tqdm(queries, desc="Processing queries", unit="query"), 1):
                query = query.strip()
                if query:
                    print(f"\n{Fore.YELLOW}Query {i}: {Style.RESET_ALL}{query}")
                    answer = self.answer_question(query)
                    print(f"{Fore.MAGENTA}Assistant: {Style.RESET_ALL}{answer}")
                    print()

        except FileNotFoundError:
            print(f"{Fore.RED}Error: The specified query file '{query_file}' was not found.")
            print(f"{Fore.YELLOW}Please make sure the file exists in the correct location.")
        except Exception as e:
            print(f"{Fore.RED}An error occurred while processing the query file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational Document Q&A Agent")
    parser.add_argument("--config", help="Path to the configuration JSON file")
    parser.add_argument("--input", help="Path to the input document or folder containing .md files")
    parser.add_argument("--query-file", help="Path to the file containing queries")
    args = parser.parse_args()

    # Default configuration
    default_config = {
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
        "include_full_content": False,
        "truncate_start": True,
        "query_expansion_output": "query_expansion.jsonl",
        "response_output": "response.jsonl",
        "api_type": "openai",
        "api_base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "query_expansion_model": "llama3.1:latest",
        "answer_generation_model": "llama3.1:latest",
        "use_conversation_history": False,
        "include_position": False,
        "max_documents_in_prompt": 3,  # Default to including 3 documents
    }

    # Load configuration from file if provided
    if args.config:
        try:
            with open(args.config, 'r') as config_file:
                file_config = json.load(config_file)
                default_config.update(file_config)
            print(f"{Fore.GREEN}Loaded configuration from {args.config}")
        except Exception as e:
            print(f"{Fore.YELLOW}Error loading config file: {str(e)}. Using default configuration.")
    else:
        print(f"{Fore.YELLOW}No config file provided. Using default configuration.")

    # Update config with command-line arguments
    if args.input:
        default_config['input'] = args.input
    if args.query_file:
        default_config['query_file'] = args.query_file

    try:
        agent = ConversationalDocQAAgent(default_config)
        agent.logger.info("ConversationalDocQAAgent initialized")

        if 'input' not in default_config or not default_config['input']:
            default_config['input'] = input("Please enter the path to the input document or folder (or press Enter to exit): ").strip()
            if not default_config['input']:
                print(f"{Fore.YELLOW}No input provided. Exiting.")
                exit(0)

        if os.path.isdir(default_config['input']):
            agent.process_folder(default_config['input'])
        elif os.path.isfile(default_config['input']):
            try:
                with open(default_config['input'], "r", encoding='utf-8') as f:
                    content = f.read()
                filename = os.path.splitext(os.path.basename(default_config['input']))[0]
                agent.add_document(content, filename, default_config['input'])
                print(f"{Fore.GREEN}Successfully added document: {filename}")
            except FileNotFoundError:
                print(f"{Fore.RED}Error: The specified file '{default_config['input']}' was not found.")
                print(f"{Fore.YELLOW}Please make sure the file exists in the correct location.")
                exit(1)
            except Exception as e:
                print(f"{Fore.RED}An error occurred while adding the document: {str(e)}")
                exit(1)
        else:
            print(f"{Fore.RED}Error: The specified input '{default_config['input']}' is neither a file nor a directory.")
            print(f"{Fore.YELLOW}Please provide a valid file or directory path.")
            exit(1)

        if default_config.get('query_file'):
            agent.logger.info(f"Processing queries from file: {default_config['query_file']}")
            agent.process_queries_from_file(default_config['query_file'])
        else:
            agent.logger.info("Starting interactive conversation mode")
            print(f"{Fore.GREEN}Welcome to the Conversational Document Q&A Agent!")
            print(f"{Fore.CYAN}You can start asking questions about the document(s). Type 'quit' to exit.")
            print()

            while True:
                user_input = input(f"{Fore.YELLOW}User: {Style.RESET_ALL}").strip().lower()
                if user_input == 'quit':
                    break
                elif user_input == 'toggle history':
                    agent.config['use_conversation_history'] = not agent.config.get('use_conversation_history', False)
                    state = "enabled" if agent.config['use_conversation_history'] else "disabled"
                    print(f"{Fore.GREEN}Conversation history is now {state}.")
                elif user_input == 'toggle position':
                    agent.toggle_position_inclusion()
                else:
                    answer = agent.answer_question(user_input)
                    print(f"{Fore.MAGENTA}Assistant: {Style.RESET_ALL}{answer}")
                    print()

        agent.export_to_jsonl(default_config['query_expansion_output'], default_config['response_output'])
        
        markdown_output = "conversation_output.md"
        agent.export_markdown_conversation(markdown_output)
        print(f"{Fore.GREEN}Markdown conversation exported to {markdown_output}")
        
        agent.logger.info("Application completed successfully")
        print(f"{Fore.GREEN}Goodbye!")

    except Exception as e:
        agent.logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        print(f"{Fore.RED}An unexpected error occurred: {str(e)}")
        print(f"{Fore.YELLOW}The application will now exit.")
