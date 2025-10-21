import argparse
import os
import torch
import numpy as np
import gc
import json
from tqdm import tqdm
from vllm import LLM, EngineArgs, SamplingParams
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from openai import OpenAI
from utils import truncate_text, load_dataset_samples
import fire
from llmlingua import PromptCompressor
from code_compressor import CodeCompressor
import asyncio
from itertools import cycle

class LLMGenerator:
    def __init__(self, model_name, device, **model_args):
        # Create a vllm LLM instance
        engine_args = EngineArgs(model=model_name, gpu_memory_utilization=0.8, device=device, **model_args)
        self.model = LLM(**vars(engine_args))
        self.model_name = model_name
        self.device = device
        # Use the tokenizer from the model to ensure consistency
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate(self, prompt, max_tokens=2048, temperature=0.0):
        logger.debug(f"Generation input prompt: {truncate_text(prompt)}")
        
        # Convert to chat format
        conversation = [
            {"role": "system", "content": "You are a documentation generating assistant specialized in code understanding."},
            {"role": "user", "content": prompt}
        ]
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=50,
        )
        
        outputs = self.model.chat(conversation, sampling_params, use_tqdm=False)
        result = outputs[0].outputs[0].text
        
        logger.debug(f"Generation output: {truncate_text(result)}")
        return result
    
    def free_memory(self):
        """Release model resources to free GPU memory"""
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


class LLMScorer:
    def __init__(self, model_name, device, **model_args):
        # Create a vllm LLM instance
        engine_args = EngineArgs(model=model_name, gpu_memory_utilization=0.8, device=device, **model_args)
        self.model = LLM(**vars(engine_args))
        self.model_name = model_name
        self.device = device
        # Use the tokenizer from the model to ensure consistency
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def score_options(self, query, options):
        # Convert to a chat format query
        conversation = [
            {"role": "system", "content": "You are a code quality assessing engine."},
            {"role": "user", "content": query}
        ]
        
        logger.debug(f"Scoring input query: {truncate_text(query)}")
        logger.debug(f"Scoring options: {options}")
        
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.3,
            logprobs=20,
        )
        
        # Get the completion with logprobs
        outputs = self.model.chat(conversation, sampling_params, use_tqdm=False)
        output = outputs[0].outputs[0]
        
        # Debug output structure
        logger.debug(f"Output structure: {type(output)}")
        logger.debug(f"Logprobs structure: {type(output.logprobs)}")
        
        # Extract logprobs for the options
        logprobs = torch.zeros(len(options))
        found_options = set()
        
        # Convert options to lowercase for case-insensitive matching
        option_map = {opt.lower(): i for i, opt in enumerate(options)}
        
        # Extract logprobs from the output
        for token_dict in output.logprobs:
            # Each item is a dictionary with token_id -> Logprob object
            for _, logprob_obj in token_dict.items():
                try:
                    # Directly access the token and logprob attributes
                    token = logprob_obj.decoded_token.strip().lower()
                    logprob_value = logprob_obj.logprob
                    
                    # Check if this token matches one of our options
                    if token in option_map and option_map[token] not in found_options:
                        logprobs[option_map[token]] = logprob_value
                        found_options.add(option_map[token])
                        logger.debug(f"Found option: {token} with logprob: {logprob_value}")
                
                except AttributeError:
                    # If the object doesn't have the expected attributes, skip it
                    continue
                except Exception as e:
                    logger.error(f"Error processing token: {e}")
                    continue
        
        # Special case for options A and B
        if not found_options and len(output.logprobs) > 0:
            for token_dict in output.logprobs:
                for _, logprob_obj in token_dict.items():
                    try:
                        # Check specifically for A or B tokens
                        token = logprob_obj.decoded_token.strip().lower()
                        
                        if token in ['a', 'b'] and option_map.get(token) not in found_options:
                            logprobs[option_map[token]] = logprob_obj.logprob
                            found_options.add(option_map[token])
                            logger.debug(f"Found exact option: {token.upper()} with logprob: {logprob_obj.logprob}")
                    except Exception as e:
                        logger.error(f"Error processing token for A/B check: {e}")
                        continue
        
        # If some options weren't found, assign a very low logprob
        min_prob = logprobs[list(found_options)].min().item() if found_options else -100
        for i in range(len(options)):
            if i not in found_options:
                logprobs[i] = min_prob - 2.3  # approximately 10 times less
        
        logger.debug(f"Final scoring output logprobs: {logprobs}")
        
        return logprobs
    
    def free_memory(self):
        """Release model resources to free GPU memory"""
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


class GPTScorer:
    def __init__(self, model_name="gpt-4o-mini", **model_args):
        self.model_name = model_name
        # Use transformers tokenizer instead of tiktoken
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using gpt2 tokenizer as a good approximation
        
        # Array of API tokens for rotation
        self.api_tokens = [
            "your_api_key"
        ]
        self.token_iterator = cycle(self.api_tokens)
        
        # Initialize OpenAI client with the first token
        self.current_token = next(self.token_iterator)
        self.client = OpenAI(
            api_key=self.current_token
        )
        logger.debug(f"Initialized GPTScorer with model: {model_name}")
    
    def rotate_token(self):
        """Rotate to the next API token"""
        self.current_token = next(self.token_iterator)
        self.client = OpenAI(
            api_key=self.current_token
        )
        logger.debug(f"Rotated to next API token")
    
    def score_options(self, query, options):
        logger.debug(f"Scoring input query: {truncate_text(query)}")
        logger.debug(f"Scoring options: {options}")
        
        # Create logit bias to prioritize the option tokens
        logit_bias = dict()
        for opt in options:
            # Use transformers tokenizer
            tok_ids = self.tokenizer.encode(opt, add_special_tokens=False)
            if len(tok_ids) == 1:
                logit_bias[tok_ids[0]] = 100
            else:
                logger.warning(f"Option '{opt}' encodes to multiple tokens {tok_ids}, using first token only")
                logit_bias[tok_ids[0]] = 100
        
        # Try up to 3 times with token rotation on failure
        for attempt in range(3):
            try:
                # Call the OpenAI API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a code quality assessing engine."},
                        {"role": "user", "content": query},
                    ],
                    max_tokens=1,
                    temperature=0.3,
                    n=1,
                    logprobs=True,
                    top_logprobs=20,
                    logit_bias=logit_bias
                )
                
                # Process the results
                logprobs = np.full(len(options), np.nan)
                choice = completion.choices[0]
                logger.debug(f"Choice: {choice}")
                opt_to_idx = {t: n for n, t in enumerate(options)}
                min_lp = 0
                
                try:
                    for logprob_item in choice.logprobs.content[0].top_logprobs:
                        tok = logprob_item.token
                        lp = logprob_item.logprob
                        min_lp = min(min_lp, lp)
                        if tok in opt_to_idx:
                            logprobs[opt_to_idx[tok]] = lp
                    
                    # If any options weren't found, assign them a low probability
                    logprobs[np.isnan(logprobs)] = min_lp - 2.3
                    assert not np.isnan(logprobs).any()
                    break  # Success, exit retry loop
                except Exception as e:
                    logger.error(f"Error processing logprobs: {e}")
                    # Return equal logprobs in case of error
                    return torch.zeros(len(options))
                    
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}/3): {e}")
                # Rotate token on failure
                self.rotate_token()
                if attempt == 2:  # Last attempt failed
                    logger.error("All API attempts failed")
                    return torch.zeros(len(options))
        
        logger.debug(f"Final scoring output logprobs: {logprobs}")
        return torch.from_numpy(logprobs)
    
    async def async_score_options(self, query, options):
        """Asynchronous version of score_options that runs in a thread pool"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.score_options, query, options
        )
    
    def free_memory(self):
        """Release any resources"""
        # Nothing to free for API-based model
        pass


# Helper function for sliding window chunking
def chunk_sliding_window(code: str, window_size: int, overlap: int) -> list[str]:
    """Splits code into overlapping chunks using a sliding window."""
    lines = code.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError("Overlap size must be smaller than window size.")

    while True:
        end = min(start + window_size, len(lines))
        chunk_lines = lines[start:end]
        if not chunk_lines:  # Should not happen if lines is not empty, but safety check
            break
        chunks.append("\n".join(chunk_lines))
        if end == len(lines):
            break  # Exit loop if we reached the end
        next_start = start + stride
        # If the next window would go past the end, break
        if next_start >= len(lines):
            # Add the final overlapping chunk if needed
            final_start = max(0, len(lines) - window_size)
            if final_start > start:  # Ensure it's a new chunk not already added
                final_chunk_lines = lines[final_start:]
                chunks.append("\n".join(final_chunk_lines))
            break
        start = next_start

    # Handle case where code is shorter than window size
    if not chunks and lines:
        return ["\n".join(lines)]

    # Remove duplicates while preserving order (important for RAG)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    return unique_chunks


# Helper function to compute embeddings (using mean pooling)
def compute_embedding(text: str, model, tokenizer, device) -> torch.Tensor:
    """Computes sentence embedding for a text using the provided model."""
    if not text.strip():  # Handle empty strings
        return torch.zeros(model.config.hidden_size).to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool the last hidden state
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding


# Helper function for RAG retrieval
def rag_retrieve(background_code: str, query_code: str, model, tokenizer, device, window_size: int, overlap: int, top_k: int) -> str:
    """Chunks background, embeds chunks and query, retrieves top_k similar chunks."""
    if not background_code.strip():
        return ""  # Return empty if no background context

    chunks = chunk_sliding_window(background_code, window_size, overlap)
    if not chunks:
        return ""  # Return empty if chunking results in nothing

    query_embedding = compute_embedding(query_code, model, tokenizer, device)

    chunk_embeddings = []
    valid_chunks = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embeddings.append(compute_embedding(chunk, model, tokenizer, device))
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ""

    # Stack embeddings for efficient similarity calculation
    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    # Compute cosine similarity
    similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    # Get top_k indices
    top_k_indices = torch.topk(similarities, k=min(top_k, len(valid_chunks)), dim=0).indices

    # Retrieve and sort chunks by their original position
    relevant_chunks_with_indices = []
    original_indices_map = {chunk_content: idx for idx, chunk_content in enumerate(chunks)}  # Map content back to original index

    retrieved_chunk_contents = [valid_chunks[i] for i in top_k_indices.tolist()]

    # Find original start lines to sort chronologically (approximate)
    chunk_start_lines = {}
    current_line = 0
    lines = background_code.splitlines()
    chunk_map_from_sliding = chunk_sliding_window(background_code, window_size, overlap)  # Re-chunk to get consistent indexing if needed
    start_line_num = 0
    stride = window_size - overlap
    for i, chunk_content in enumerate(chunk_map_from_sliding):
        # This assumes the chunking function returns chunks in order
        chunk_start_lines[chunk_content] = start_line_num
        start_line_num += stride
        # Rough approximation, doesn't perfectly handle edge cases/final chunks

    sorted_relevant_chunks = sorted(
        retrieved_chunk_contents,
        key=lambda content: chunk_start_lines.get(content, float('inf'))  # Sort by approximate start line
    )

    # Combine relevant chunks
    # Original implementation joined with \n, let's keep it simple
    combined_code = "\n\n".join(sorted_relevant_chunks)  # Separate chunks by double newline for clarity

    return combined_code


# Helper function for LLMLingua compression
def compress_llmlingua(context: str, query: str, compressor: PromptCompressor, target_token: int, instruction: str) -> str:
    """Compresses context using LLMLingua."""
    if not context.strip():
        return ""
    try:
        # Ensure no "<|endoftext|>"
        context_clean = context.replace("<|endoftext|>", "")
        compressed = compressor.compress_prompt(
            context_clean,
            instruction=instruction,
            question=query + "\n" + instruction,  # Combine query and instruction for question
            target_token=target_token
        )
        # Ensure result exists and is string
        result = compressed.get('compressed_prompt', '')
        return result if isinstance(result, str) else ""
    except Exception as e:
        logger.error(f"LLMLingua compression failed: {e}")
        # Fallback: Truncate based on target tokens (approximate)
        tokens = compressor.tokenizer.encode(context_clean)
        if len(tokens) > target_token:
            return compressor.tokenizer.decode(tokens[:target_token])
        return context_clean


# Helper function for LongLLMLingua compression
def compress_longllmlingua(context: str, query: str, compressor: PromptCompressor, target_token: int, instruction: str, chunk_size: int, overlap: int) -> str:
    """Compresses context using LongLLMLingua with sliding window chunks."""
    if not context.strip():
        return ""
    try:
        # Ensure no "<|endoftext|>"
        context_clean = context.replace("<|endoftext|>", "")
        # Use our sliding window chunker
        chunks = chunk_sliding_window(context_clean, chunk_size, overlap)
        if not chunks:
            return ""  # Handle case where context is too short or chunking fails

        compressed = compressor.compress_prompt(
            chunks,
            instruction=instruction,
            question=query + "\n" + instruction,  # Combine query and instruction for question
            target_token=target_token,
            rank_method="longllmlingua"  # Use the specified rank method
        )
        # Ensure result exists and is string
        result = compressed.get('compressed_prompt', '')
        return result if isinstance(result, str) else ""
    except Exception as e:
        logger.error(f"LongLLMLingua compression failed: {e}")
        # Fallback: Truncate based on target tokens (approximate)
        tokens = compressor.tokenizer.encode(context_clean)
        if len(tokens) > target_token:
            return compressor.tokenizer.decode(tokens[:target_token])
        return context_clean


# Helper function for CodeCompressor
def compress_code_compressor(context: str, query: str, compressor, target_token: int, instruction: str, language: str, rank_only: bool, fine_ratio: float, importance_beta: float) -> str:
    """Compresses context using CodeCompressor based on target tokens and rank_only flag."""
    if not context.strip():
        return ""
    try:
        # Ensure no "<|endoftext|>"
        context_clean = context.replace("<|endoftext|>", "")
        if not context_clean.strip():
            return ""  # Return empty if clean context is empty

        # Tokenize to get original length
        # Use the compressor's tokenizer
        original_tokens = len(compressor.tokenizer.encode(context_clean))
        if original_tokens == 0:
            return ""  # Avoid division by zero

        # Calculate target ratio
        target_ratio = min(1.0, max(0.0, target_token / original_tokens))
        logger.info(f"CodeCompressor: Original tokens={original_tokens}, Target tokens={target_token}, Calculated ratio={target_ratio:.4f}")

        # Pass rank_only and fine_ratio
        # Assuming compressor is already initialized with the correct model
        compressed_result = compressor.compress_code_file(
            code=context_clean,
            query=query,  # Using current function context as query focus
            instruction=instruction,
            rate=target_ratio,
            language=language,
            rank_only=rank_only,  # Ensure rank_only mode is set
            fine_ratio=fine_ratio if not rank_only else None,  # Pass fine_ratio only if not rank_only
            importance_beta=importance_beta if not rank_only else None,  # Pass importance_beta only if not rank_only
        )

        # Extract compressed content - check both possible keys
        compressed_context = compressed_result.get("compressed_code")

        if not isinstance(compressed_context, str):
            logger.error(f"CodeCompressor returned non-string: {type(compressed_context)}")
            compressed_context = ""  # Fallback

        # Log results
        compressed_tokens_count = len(compressor.tokenizer.encode(compressed_context))
        final_ratio = (compressed_tokens_count / original_tokens) if original_tokens > 0 else 0
        logger.info(f"CodeCompressor: Compressed tokens={compressed_tokens_count}, Actual ratio={final_ratio:.4f}")

        return compressed_context

    except Exception as e:
        logger.error(f"CodeCompressor compression failed: {e}", exc_info=True)
        # Fallback: Truncate approximately based on target tokens (less ideal for rank_only)
        tokens = compressor.tokenizer.encode(context_clean)
        if len(tokens) > target_token:
            logger.warning(f"CodeCompressor falling back to simple truncation.")
            return compressor.tokenizer.decode(tokens[:target_token])
        return context_clean


# Helper function for splitting code by functions (from main_lcc.py)
def split_code_by_functions_standalone(code: str, language: str = "python") -> list[str]:
    """
    Split code into chunks based on function and class definitions for various languages.
    Standalone version that doesn't require CodeCompressor instance.
    
    Args:
        code: The code to split
        language: Programming language of the code (python, cpp, java, typescript, rust, go)
        
    Returns:
        List of code chunks, each containing a function, class, or class method
    """
    import re
    
    # Define regex patterns for different languages
    patterns = {
        # Python: Simplified to match 'def' or 'class' followed by content until the next def/class or end
        "python": r'(^|\n)(\s*)(def|class)\s+[^\n]+(\n(?!\s*(?:def|class)\s)[^\n]*)*',
        # C++: Improved to better handle multi-line declarations
        "cpp": r'(^|\n)(\s*)(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*[^{]*)?|(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Java: Improved for multi-line method declarations
        "java": r'(^|\n)(\s*)(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:<.*>)?(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*throws\s+[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # TypeScript: Enhanced to handle multi-line methods and arrow functions
        "typescript": r'(^|\n)(\s*)(?:(?:public|private|protected|static|abstract)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:(?:public|private|protected|static|async)\s+)*(?:function\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\([^{;]*\)\s*(?::\s*[^{;]*\s*)?(?:=>)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Rust: Improved for multi-line function declarations
        "rust": r'(^|\n)(\s*)(?:pub\s+)?(?:struct\s+[a-zA-Z_][a-zA-Z0-9_]*|impl(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+for\s+[a-zA-Z_][a-zA-Z0-9_]*)?|(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<.*>)?\s*\([^{;]*\)(?:\s*->\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Go: Improved for multi-line function declarations
        "go": r'(^|\n)(\s*)(?:type\s+[a-zA-Z_][a-zA-Z0-9_]*\s+struct|func\s+(?:\([^)]*\)\s*)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
    }
    
    # Use default Python pattern if language not supported
    if language.lower() not in patterns:
        language = "python"
    
    function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
    matches = list(function_pattern.finditer(code))
    
    if not matches:
        return [code] if code.strip() else []  # No matches, return whole code if not empty
        
    result_chunks = []
    
    # Add code before first match if exists
    if matches[0].start() > 0:
        pre_code = code[:matches[0].start()].strip()
        if pre_code:
            result_chunks.append(pre_code)
    
    # Process each match
    for i, match in enumerate(matches):
        start = match.start()
        
        # End is either start of next match or end of code
        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(code)
        
        chunk = code[start:end].strip()
        if chunk:
            result_chunks.append(chunk)
    
    return result_chunks


# Helper function for function-level RAG retrieval with budget
def function_rag_retrieve(background_code: str, query_code: str, model, tokenizer, device, language: str, budget: int) -> str:
    """Uses function-level chunking and retrieves functions within the specified token budget."""
    if not background_code.strip():
        return ""  # Return empty if no background context

    # Split code into function-based chunks
    chunks = split_code_by_functions_standalone(background_code, language)
    if not chunks:
        return ""  # Return empty if chunking results in nothing

    query_embedding = compute_embedding(query_code, model, tokenizer, device)

    chunk_embeddings = []
    valid_chunks = []
    for chunk in chunks:
        if chunk.strip():
            chunk_embeddings.append(compute_embedding(chunk, model, tokenizer, device))
            valid_chunks.append(chunk)

    if not valid_chunks:
        return ""

    # Stack embeddings for efficient similarity calculation
    chunk_embeddings_tensor = torch.stack(chunk_embeddings)

    # Compute cosine similarity
    similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings_tensor, dim=1)

    # Sort chunks by similarity score (descending)
    sorted_indices = torch.argsort(similarities, descending=True)

    # Select chunks within budget
    selected_chunks = []
    current_tokens = 0
    
    for idx in sorted_indices:
        chunk = valid_chunks[idx.item()]
        
        # Calculate tokens for this chunk
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        
        # Check if adding this chunk would exceed budget
        if current_tokens + chunk_tokens <= budget:
            selected_chunks.append((chunk, similarities[idx].item()))
            current_tokens += chunk_tokens
        else:
            # Try to partially include the chunk if there's remaining budget
            remaining_budget = budget - current_tokens
            if remaining_budget > 50:  # Only include if we have at least 50 tokens left
                chunk_tokens_list = tokenizer.encode(chunk, add_special_tokens=False)
                if len(chunk_tokens_list) > remaining_budget:
                    # Truncate the chunk to fit the remaining budget
                    truncated_tokens = chunk_tokens_list[:remaining_budget]
                    truncated_chunk = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    selected_chunks.append((truncated_chunk, similarities[idx].item()))
                    current_tokens = budget
            break
        
        # Stop if we've reached the budget
        if current_tokens >= budget:
            break

    if not selected_chunks:
        return ""

    # Sort selected chunks by their original position in the code to maintain structure
    # We'll use the similarity score for this approximation since we don't have direct position info
    selected_chunks.sort(key=lambda x: x[1], reverse=True)  # Keep similarity order for now
    
    # Combine selected chunks
    combined_code = "\n\n".join([chunk for chunk, _ in selected_chunks])
    
    logger.info(f"Function RAG: Selected {len(selected_chunks)} functions using {current_tokens}/{budget} tokens")
    
    return combined_code


async def async_get_metric(scorer, intent, code_context, gold_doc, pred_doc):
    logger.debug(f"Evaluating intent: {intent}")
    logger.debug(f"Gold doc: {truncate_text(gold_doc)}")
    logger.debug(f"Pred doc: {truncate_text(pred_doc)}")
    logger.debug(f"Gold doc length: {len(gold_doc)}, Pred doc length: {len(pred_doc)}")
    
    prompt = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n' 
    prompt += f'My code:\n\n{code_context}\n\n\n\n'
    prompt += f'Documentation A:\n\n{gold_doc}\n\n\n\n'
    prompt += f'Documentation B:\n\n{pred_doc}\n\n\n\n'
    prompt += 'Please directly return the option that is better (A or B) without any other text.'
    
    options = ["A", "B"]
    unnorm_logprobs = await scorer.async_score_options(prompt, options)
    norm_probs1 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))
    
    prompt = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n' 
    prompt += f'My code:\n\n{code_context}\n\n\n\n'
    prompt += f'Documentation A:\n\n{pred_doc}\n\n\n\n'
    prompt += f'Documentation B:\n\n{gold_doc}\n\n\n\n'
    prompt += 'Please directly return the option that is better (A or B) without any other text.'
    unnorm_logprobs = await scorer.async_score_options(prompt, options)
    norm_probs2 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))
    
    p_better1 = (norm_probs1[1] + norm_probs2[0]) / 2 
    logger.debug(f"First evaluation: {norm_probs1}, Second evaluation: {norm_probs2}, Final score: {p_better1}")
    
    return float(p_better1)


def get_metric(scorer, intent, code_context, gold_doc, pred_doc):
    logger.debug(f"Evaluating intent: {intent}")
    logger.debug(f"Gold doc: {truncate_text(gold_doc)}")
    logger.debug(f"Pred doc: {truncate_text(pred_doc)}")
    logger.debug(f"Gold doc length: {len(gold_doc)}, Pred doc length: {len(pred_doc)}")
    
    prompt = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n' 
    prompt += f'My code:\n\n{code_context}\n\n\n\n'
    prompt += f'Documentation A:\n\n{gold_doc}\n\n\n\n'
    prompt += f'Documentation B:\n\n{pred_doc}\n\n\n\n'
    prompt += 'Please directly return the option that is better (A or B) without any other text.'
    
    options = ["A", "B"]
    unnorm_logprobs = scorer.score_options(prompt, options)
    norm_probs1 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))
    
    prompt = f'I have 2 different documentations about {intent}. Decide which documentation is better: documentation A or documentation B.\n\n' 
    prompt += f'My code:\n\n{code_context}\n\n\n\n'
    prompt += f'Documentation A:\n\n{pred_doc}\n\n\n\n'
    prompt += f'Documentation B:\n\n{gold_doc}\n\n\n\n'
    prompt += 'Please directly return the option that is better (A or B) without any other text.'
    unnorm_logprobs = scorer.score_options(prompt, options)
    norm_probs2 = torch.exp(torch.log_softmax(unnorm_logprobs, dim=0))
    
    p_better1 = (norm_probs1[1] + norm_probs2[0]) / 2 
    logger.debug(f"First evaluation: {norm_probs1}, Second evaluation: {norm_probs2}, Final score: {p_better1}")
    
    return float(p_better1)


async def evaluate_batch(batch_data, scorer, samples_data, method, is_async=True):
    """Evaluate a batch of samples"""
    results = []
    
    for item in batch_data:
        idx, row = item
        gold_doc = row['target_text']
        
        # Skip if sample data doesn't exist
        if idx >= len(samples_data) or not samples_data[idx]:
            logger.warning(f"Sample data not found for sample {idx}. Skipping evaluation.")
            continue
            
        # Get sample data
        sample_data = samples_data[idx]
        pred_doc = sample_data.get('generated_text', '')
        
        code_context = row['relevant_code_context']
        
        # Use the appropriate metric function based on whether the scorer is async
        if is_async:
            metric = await async_get_metric(scorer, row['intent'], code_context, gold_doc, pred_doc)
        else:
            # For synchronous scorers, run in an executor to not block the event loop
            loop = asyncio.get_running_loop()
            metric = await loop.run_in_executor(
                None, get_metric, scorer, row['intent'], code_context, gold_doc, pred_doc
            )
        
        # Update sample data with evaluation score
        sample_data['generation_score'] = float(metric)
        
        results.append((idx, metric, sample_data))
    
    return results


async def run_parallel_evaluation(dataset, scorer, samples_data, method, num_processes=4, is_async=True):
    """Run evaluation in parallel using specified number of processes"""
    # Prepare data with indices
    indexed_data = list(enumerate(dataset))
    
    # Split data into chunks for each process
    chunk_size = len(indexed_data) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    
    batches = [indexed_data[i:i+chunk_size] for i in range(0, len(indexed_data), chunk_size)]
    
    # Ensure we don't create more batches than needed
    batches = batches[:num_processes]
    
    # Create tasks for each batch
    tasks = [evaluate_batch(batch, scorer, samples_data, method, is_async) for batch in batches]
    
    # Run all batches concurrently and collect results
    batch_results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_results = []
    for batch in batch_results:
        all_results.extend(batch)
    
    # Sort by sample index
    all_results.sort(key=lambda x: x[0])
    
    # Extract metrics and metadata
    metrics = [r[1] for r in all_results]
    detailed_results = [r[2] for r in all_results]
    
    return metrics, detailed_results


def run_documentation_task(
    # Generation model parameters
    gen_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    compress_model: str = None,
    model_name: str = None,
    # Evaluation model parameters
    eval_model: str = "gpt-4o-mini",
    # Common model parameters
    device: str = "cuda",
    tensor_parallel_size: int = 1,
    # Generation parameters
    max_tokens: int = 2048,
    temperature: float = 0.0,
    # Context method parameters
    method: str = "full",
    # RAG parameters
    rag_window_size: int = 80,
    rag_overlap: int = 40,
    rag_top_k: int = 3,
    embed_model_name: str = "microsoft/unixcoder-base",
    # Function RAG parameters
    function_rag_language: str = "python",
    function_rag_budget: int = 1024,
    # LLMLingua parameters
    lingua_target_token: int = 500,
    lingua_instruction: str = "Generate documentation based on this code.",
    # LongLLMLingua parameters
    longlingua_chunk_size: int = 80,
    longlingua_overlap: int = 40,
    # CodeCompressor parameters
    code_compressor_target_token: int = 500,
    code_compressor_fine_ratio: float = 1.0,
    importance_beta: float = 0.0,
    # Task parameters
    mode: str = "both",
    save_dir: str = "./predictions",
    hf_api_key: str = None,
    max_examples: int = None,
    use_llm_scorer: bool = False,
    # Parallel evaluation parameters
    num_eval_processes: int = 4
):
    """Run documentation generation and evaluation with the specified parameters."""
    
    # Get model short name from argument or extract from model path
    model_short_name = model_name
    if model_short_name is None:
        # Extract model name from path - use last component after / or use the whole string
        model_short_name = gen_model.split('/')[-1] if '/' in gen_model else gen_model
    
    if compress_model is None:
        compress_model = gen_model
        logger.info(f"Using generation model for compression: {compress_model}")

    # Create method-specific suffix for results directory
    method_suffix = f"method_{method}"
    if method == "rag":
        method_suffix += f"_w{rag_window_size}_o{rag_overlap}_k{rag_top_k}"
    elif method == "function_rag":
        method_suffix += f"_lang{function_rag_language}_b{function_rag_budget}"
    elif method == "llmlingua":
        method_suffix += f"_t{lingua_target_token}"
    elif method == "longllmlingua":
        method_suffix += f"_t{lingua_target_token}_cs{longlingua_chunk_size}_o{longlingua_overlap}"
    elif method == "code_compressor":
        # Determine if rank_only based on fine_ratio
        rank_only_for_suffix = (code_compressor_fine_ratio == 1.0)
        suffix_detail = "_rankonly" if rank_only_for_suffix else f"_fr{code_compressor_fine_ratio}"
        # Add importance_beta to suffix
        if importance_beta > 0:
            suffix_detail += f"_b{importance_beta}"
        method_suffix += f"_t{code_compressor_target_token}{suffix_detail}"
    
    # Create method-specific directory
    model_save_dir = os.path.join(save_dir, method_suffix, model_short_name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # Path to our single results JSON file
    results_json_path = os.path.join(model_save_dir, "detailed_results.json")
    
    # Load dataset
    print("Loading dataset")
    dataset = load_dataset_samples(
        max_examples=max_examples,
        hf_api_key=hf_api_key
    )
    
    # Common model args for both generator and scorer
    model_args = {
        "tensor_parallel_size": tensor_parallel_size,
    }

    # Initialize or load samples data
    samples_data = []
    
    # Check if results file exists (for continuing an interrupted run)
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                existing_results = json.load(f)
                # Extract existing samples data
                samples_data = existing_results.get('samples', [])
                # Ensure we have enough entries for all samples
                if len(samples_data) < len(dataset):
                    samples_data.extend([None] * (len(dataset) - len(samples_data)))
                print(f"Loaded existing results with {len(samples_data)} samples.")
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.")
            samples_data = [None] * len(dataset)
    else:
        # Initialize with empty slots for each sample
        samples_data = [None] * len(dataset)
    
    # Generation phase - split into compression and generation steps
    if mode in ["generate", "both"]:
        # Step 1: Compress contexts first if needed
        if method not in ["full", "no_context"]:
            print(f"Step 1: Preparing contexts using {method} method...")
            
            # Initialize context preparation models based on method
            embed_model = None
            embed_tokenizer = None
            lingua_compressor = None
            code_compressor_instance = None
            
            if method == "rag":
                print(f"Initializing embedding model: {embed_model_name}")
                embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
                embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
                embed_model.eval()  # Set to evaluation mode
            
            if method == "function_rag":
                print(f"Initializing embedding model for function RAG: {embed_model_name}")
                embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
                embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
                embed_model.eval()  # Set to evaluation mode
            
            if method == "llmlingua" or method == "longllmlingua":
                print(f"Initializing LLMLingua compressor")
                lingua_compressor = PromptCompressor(model_name=gen_model, device_map="auto")
            
            if method == "code_compressor":
                try:
                    print(f"Initializing CodeCompressor")
                    code_compressor_instance = CodeCompressor(gen_model)
                except Exception as e:
                    print(f"Failed to initialize CodeCompressor: {e}. Falling back to full context.")
                    method = "full"
            
            # Process and compress all contexts
            for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Compressing contexts"):
                # If sample already has context processing, skip
                if samples_data[idx] and 'processed_context' in samples_data[idx]:
                    continue
                    
                # Get the context
                code_context = row['relevant_code_context']
                
                # Process context based on the selected method
                processed_context = code_context
                language = "python"  # Default language, could be determined dynamically
                
                try:
                    if method == "rag":
                        # Split the context and retrieve relevant parts
                        background_ctx = code_context
                        query_ctx = f"Generate documentation for {row['docfile_name']} about {row['intent']}"
                        processed_context = rag_retrieve(
                            background_ctx, query_ctx,
                            embed_model, embed_tokenizer, device,
                            rag_window_size, rag_overlap, rag_top_k
                        )
                    elif method == "function_rag":
                        # Split the context and retrieve relevant functions within budget
                        background_ctx = code_context
                        query_ctx = f"Generate documentation for {row['docfile_name']} about {row['intent']}"
                        processed_context = function_rag_retrieve(
                            background_ctx, query_ctx,
                            embed_model, embed_tokenizer, device,
                            function_rag_language, function_rag_budget
                        )
                    elif method == "llmlingua":
                        background_ctx = code_context
                        query_ctx = f"Generate documentation for {row['docfile_name']} about {row['intent']}"
                        processed_context = compress_llmlingua(
                            background_ctx, query_ctx,
                            lingua_compressor, lingua_target_token, lingua_instruction
                        )
                    elif method == "longllmlingua":
                        background_ctx = code_context
                        query_ctx = f"Generate documentation for {row['docfile_name']} about {row['intent']}"
                        processed_context = compress_longllmlingua(
                            background_ctx, query_ctx,
                            lingua_compressor, lingua_target_token, lingua_instruction,
                            longlingua_chunk_size, longlingua_overlap
                        )
                    elif method == "code_compressor":
                        # Determine rank_only based on fine_ratio
                        rank_only = (code_compressor_fine_ratio == 1.0)
                        logger.info(f"CodeCompressor mode: {'Rank Only' if rank_only else f'Fine-grained (ratio={code_compressor_fine_ratio})'}")
                        background_ctx = code_context
                        query_ctx = f"Generate documentation for {row['docfile_name']} about {row['intent']}"
                        processed_context = compress_code_compressor(
                            context=background_ctx,
                            query=query_ctx,
                            compressor=code_compressor_instance,
                            target_token=code_compressor_target_token,
                            instruction=lingua_instruction,
                            language=language,
                            rank_only=rank_only,
                            fine_ratio=code_compressor_fine_ratio,
                            importance_beta=importance_beta
                        )
                except Exception as e:
                    logger.error(f"Error during context preparation with method {method}: {e}", exc_info=True)
                    # Fallback to full context in case of error
                    processed_context = code_context
                
                # Create or update sample data
                sample_data = samples_data[idx] or {}
                sample_data.update({
                    'sample_id': idx,
                    'intent': row['intent'],
                    'docfile_name': row['docfile_name'],
                    'target_text': row['target_text'],
                    'original_context': code_context,
                    'processed_context': processed_context,
                    'context_compression': {
                        'method': method,
                        'original_length': len(code_context),
                        'processed_length': len(processed_context),
                        'compression_ratio': len(processed_context) / len(code_context) if len(code_context) > 0 else 1.0,
                        'method_params': {
                            'type': method,
                            'rag_params': {
                                'window_size': rag_window_size,
                                'overlap': rag_overlap,
                                'top_k': rag_top_k
                            } if method == "rag" else None,
                            'function_rag_params': {
                                'language': function_rag_language,
                                'budget': function_rag_budget
                            } if method == "function_rag" else None,
                            'llmlingua_params': {
                                'target_token': lingua_target_token
                            } if method == "llmlingua" else None,
                            'longllmlingua_params': {
                                'target_token': lingua_target_token,
                                'chunk_size': longlingua_chunk_size,
                                'overlap': longlingua_overlap
                            } if method == "longllmlingua" else None,
                            'code_compressor_params': {
                                'target_token': code_compressor_target_token,
                                'rank_only': (code_compressor_fine_ratio == 1.0),
                                'fine_ratio': code_compressor_fine_ratio,
                                'importance_beta': importance_beta
                            } if method == "code_compressor" else None
                        }
                    }
                })
                
                # Update samples data
                samples_data[idx] = sample_data
                
                # Save the updated results file periodically
                if idx % 10 == 0 or idx == len(dataset) - 1:
                    results_data = {
                        'model': model_short_name,
                        'method': method,
                        'method_params': {
                            'type': method,
                            'rag_params': {
                                'window_size': rag_window_size,
                                'overlap': rag_overlap,
                                'top_k': rag_top_k
                            } if method == "rag" else None,
                            'function_rag_params': {
                                'language': function_rag_language,
                                'budget': function_rag_budget
                            } if method == "function_rag" else None,
                            'llmlingua_params': {
                                'target_token': lingua_target_token
                            } if method == "llmlingua" else None,
                            'longllmlingua_params': {
                                'target_token': lingua_target_token,
                                'chunk_size': longlingua_chunk_size,
                                'overlap': longlingua_overlap
                            } if method == "longllmlingua" else None,
                            'code_compressor_params': {
                                'target_token': code_compressor_target_token,
                                'rank_only': (code_compressor_fine_ratio == 1.0),
                                'fine_ratio': code_compressor_fine_ratio,
                                'importance_beta': importance_beta
                            } if method == "code_compressor" else None
                        },
                        'average_score': None,  # Will be filled during evaluation
                        'samples': samples_data
                    }
                    # make sure the results_json_path is a valid path
                    if not os.path.exists(os.path.dirname(results_json_path)):
                        os.makedirs(os.path.dirname(results_json_path))
                    with open(results_json_path, 'w') as f:
                        json.dump(results_data, f, indent=2)
            
            # Free up context preparation resources
            print("Cleaning up context preparation resources...")
            if embed_model:
                del embed_model
            if embed_tokenizer:
                del embed_tokenizer
            if lingua_compressor:
                del lingua_compressor
            if code_compressor_instance:
                del code_compressor_instance
            torch.cuda.empty_cache()
            gc.collect()
        
        # Step 2: Generate documentation using the (potentially compressed) contexts
        print(f"Step 2: Initializing generation model: {gen_model}")
        generator = LLMGenerator(gen_model, device, **model_args)
        
        # Define a token limit for context when method is "full"
        MAX_CONTEXT_TOKENS_FOR_FULL_METHOD = 30000

        print(f"Generating documentation...")
        for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Generating documentation"):
            # Skip if this sample already has generated text
            if samples_data[idx] and 'generated_text' in samples_data[idx]:
                continue
                
            # Create or load sample data
            sample_data = samples_data[idx] or {}
            
            # Determine context to use
            if method not in ["full", "no_context"] and sample_data.get('processed_context'):
                context = sample_data.get('processed_context')
            else:
                # For full or no_context methods
                context = row['relevant_code_context']
                
                # For no_context, use minimal information
                if method == "no_context":
                    context = f"Generate documentation for {row['docfile_name']} about {row['intent']}"
                
                # Update sample data with context info if not already there
                if 'original_context' not in sample_data:
                    sample_data.update({
                        'sample_id': idx,
                        'intent': row['intent'],
                        'docfile_name': row['docfile_name'],
                        'target_text': row['target_text'],
                        'original_context': row['relevant_code_context'],
                        'processed_context': None if method == "full" else context
                    })
            
            # Truncate context if method is "full" and context is too long
            if method == "full":
                context_tokens = generator.tokenizer.encode(context)
                if len(context_tokens) > MAX_CONTEXT_TOKENS_FOR_FULL_METHOD:
                    logger.warning(f"Sample {idx}: Context for 'full' method was too long ({len(context_tokens)} tokens). Truncating to {MAX_CONTEXT_TOKENS_FOR_FULL_METHOD} tokens.")
                    truncated_tokens = context_tokens[:MAX_CONTEXT_TOKENS_FOR_FULL_METHOD]
                    context = generator.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    # Update context length in sample_data if it was already populated
                    if 'context_length' in sample_data:
                        sample_data['context_length'] = len(context)

            # Generate documentation
            prompt = f'Using the code provided, generate documentation for {row["docfile_name"]} about {row["intent"]}.\n\n'
            prompt += f'Code:\n\n{context}'
            prompt += f'\n\n\nReturn only the documentation text for {row["docfile_name"]} about {row["intent"]}. Do not include instructions or explanations.'
            
            generated_doc = generator.generate(prompt, max_tokens, temperature)
            
            # Update sample data with generated text
            sample_data.update({
                'generated_text': generated_doc,
                'target_text_length': len(row['target_text']),
                'generated_text_length': len(generated_doc),
                'context_length': len(context),
                'method': method,
                'method_params': {
                    'type': method,
                    'rag_params': {
                        'window_size': rag_window_size,
                        'overlap': rag_overlap,
                        'top_k': rag_top_k
                    } if method == "rag" else None,
                    'function_rag_params': {
                        'language': function_rag_language,
                        'budget': function_rag_budget
                    } if method == "function_rag" else None,
                    'llmlingua_params': {
                        'target_token': lingua_target_token
                    } if method == "llmlingua" else None,
                    'longllmlingua_params': {
                        'target_token': lingua_target_token,
                        'chunk_size': longlingua_chunk_size,
                        'overlap': longlingua_overlap
                    } if method == "longllmlingua" else None,
                    'code_compressor_params': {
                        'target_token': code_compressor_target_token,
                        'rank_only': (code_compressor_fine_ratio == 1.0),
                        'fine_ratio': code_compressor_fine_ratio,
                        'importance_beta': importance_beta
                    } if method == "code_compressor" else None
                }
            })
            
            # Update samples data
            samples_data[idx] = sample_data
            
            # Save the updated results file periodically
            if idx % 10 == 0 or idx == len(dataset) - 1:
                results_data = {
                    'model': model_short_name,
                    'method': method,
                    'method_params': {
                        'type': method,
                        'rag_params': {
                            'window_size': rag_window_size,
                            'overlap': rag_overlap,
                            'top_k': rag_top_k
                        } if method == "rag" else None,
                        'function_rag_params': {
                            'language': function_rag_language,
                            'budget': function_rag_budget
                        } if method == "function_rag" else None,
                        'llmlingua_params': {
                            'target_token': lingua_target_token
                        } if method == "llmlingua" else None,
                        'longllmlingua_params': {
                            'target_token': lingua_target_token,
                            'chunk_size': longlingua_chunk_size,
                            'overlap': longlingua_overlap
                        } if method == "longllmlingua" else None,
                        'code_compressor_params': {
                            'target_token': code_compressor_target_token,
                            'rank_only': (code_compressor_fine_ratio == 1.0),
                            'fine_ratio': code_compressor_fine_ratio,
                            'importance_beta': importance_beta
                        } if method == "code_compressor" else None
                    },
                    'average_score': None,  # Will be filled during evaluation
                    'samples': samples_data
                }
                # make sure the results_json_path is a valid path
                if not os.path.exists(os.path.dirname(results_json_path)):
                    os.makedirs(os.path.dirname(results_json_path))
                with open(results_json_path, 'w') as f:
                    json.dump(results_data, f, indent=2)
        
        # Free up memory after generation
        print("Freeing generator memory...")
        generator.free_memory()
        del generator
        torch.cuda.empty_cache()
        gc.collect()
    
    # Evaluation phase
    if mode in ["evaluate", "both"]:
        # Initialize the scorer based on the model type
        if use_llm_scorer:
            print(f"Initializing LLM evaluation model: {eval_model}")
            scorer = LLMScorer(eval_model, device, **model_args)
            is_async = False
        else:
            print(f"Initializing GPT evaluation model: {eval_model}")
            scorer = GPTScorer(eval_model, **model_args)
            is_async = True
        
        print(f"Evaluating documentation with {num_eval_processes} parallel processes...")
        
        # Use asyncio to run evaluation in parallel
        metrics, detailed_results = asyncio.run(
            run_parallel_evaluation(dataset, scorer, samples_data, method, num_eval_processes, is_async)
        )
        
        # Update samples data with evaluation scores
        for idx, result in enumerate(detailed_results):
            if idx < len(samples_data) and samples_data[idx]:
                # Update with evaluation score
                samples_data[idx]['generation_score'] = result.get('generation_score')
        
        average_metric = np.mean([s.get('generation_score', 0) for s in samples_data if s and 'generation_score' in s])
        print(f"Average evaluation metric: {average_metric:.4f}")
        
        # Save evaluation results
        if not os.path.exists(os.path.dirname(results_json_path)):
            os.makedirs(os.path.dirname(results_json_path))
        with open(os.path.join(model_save_dir, "metrics.txt"), 'w') as f:
            f.write(f"Average metric: {average_metric:.4f}\n")
            f.write("Individual metrics:\n")
            for idx, sample in enumerate(samples_data):
                if sample and 'generation_score' in sample:
                    f.write(f"Sample {idx}: {sample['generation_score']:.4f}\n")
        
        # Save final detailed results
        if not os.path.exists(os.path.dirname(results_json_path)):
            os.makedirs(os.path.dirname(results_json_path))
        with open(results_json_path, 'w') as f:
            results_data = {
                'model': model_short_name,
                'method': method,
                'method_params': {
                    'type': method,
                    'rag_params': {
                        'window_size': rag_window_size,
                        'overlap': rag_overlap,
                        'top_k': rag_top_k
                    } if method == "rag" else None,
                    'function_rag_params': {
                        'language': function_rag_language,
                        'budget': function_rag_budget
                    } if method == "function_rag" else None,
                    'llmlingua_params': {
                        'target_token': lingua_target_token
                    } if method == "llmlingua" else None,
                    'longllmlingua_params': {
                        'target_token': lingua_target_token,
                        'chunk_size': longlingua_chunk_size,
                        'overlap': longlingua_overlap
                    } if method == "longllmlingua" else None,
                    'code_compressor_params': {
                        'target_token': code_compressor_target_token,
                        'rank_only': (code_compressor_fine_ratio == 1.0),
                        'fine_ratio': code_compressor_fine_ratio,
                        'importance_beta': importance_beta
                    } if method == "code_compressor" else None
                },
                'average_score': float(average_metric),
                'samples': samples_data
            }
            json.dump(results_data, f, indent=2)
        
        # Free up scorer memory
        scorer.free_memory()


if __name__ == "__main__":

    fire.Fire(run_documentation_task)
