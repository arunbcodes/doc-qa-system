# RAGInterface API

Retrieval-Augmented Generation system combining document retrieval with LLM generation.

## Class: RAGInterface

::: src.rag.RAGInterface

### Overview

The `RAGInterface` class implements a complete RAG (Retrieval-Augmented Generation) pipeline that:

1. Retrieves relevant document chunks using semantic search
2. Constructs context-aware prompts
3. Generates answers using an LLM

The system is **model-agnostic** and works with any LLM provider.

## Basic Usage

```python
from src import RAGInterface, EmbeddingModel, VectorStore
from src.llm_providers import get_available_llm

# Initialize components
embedder = EmbeddingModel()
store = VectorStore(collection_name="docs", persist_directory="./chroma_db")
llm = get_available_llm()

# Create RAG interface
rag = RAGInterface(
    embedding_model=embedder,
    vector_store=store,
    llm=llm,
    n_results=3
)

# Ask questions
result = rag.answer_question("What is machine learning?")
print(result["answer"])
```

## Methods

### `__init__(embedding_model, vector_store, llm=None, n_results=3)`

Initialize RAG interface.

**Parameters:**

- `embedding_model` (EmbeddingModel): Model for creating embeddings
- `vector_store` (VectorStore): Vector database with document chunks
- `llm` (Optional[BaseLLM]): LLM provider (auto-detects if None)
- `n_results` (int): Number of chunks to retrieve for context (default: 3)

**Example:**

```python
from src import RAGInterface, EmbeddingModel, VectorStore
from src.llm_providers import OpenAILLM

embedder = EmbeddingModel()
store = VectorStore(collection_name="docs")
llm = OpenAILLM(model="gpt-3.5-turbo")

rag = RAGInterface(
    embedding_model=embedder,
    vector_store=store,
    llm=llm,
    n_results=5  # Retrieve more context
)
```

### `answer_question(question, temperature=0.7, max_tokens=500, show_context=False) -> Dict`

Answer a question using RAG.

**Parameters:**

- `question` (str): User's question
- `temperature` (float): LLM temperature 0.0-1.0 (default: 0.7)
- `max_tokens` (int): Maximum tokens in response (default: 500)
- `show_context` (bool): Include retrieved chunks in result (default: False)

**Returns:**

- `Dict`: Dictionary containing:
  - `answer` (str): Generated answer
  - `question` (str): Original question
  - `context` (List[Dict], optional): Retrieved chunks if `show_context=True`
  - `prompt` (str, optional): Full prompt if `show_context=True`

**Example:**

```python
# Basic answer
result = rag.answer_question("What is Python?")
print(result["answer"])

# With context
result = rag.answer_question(
    "What is Python?",
    temperature=0.5,
    max_tokens=200,
    show_context=True
)

print("Answer:", result["answer"])
print("\nRetrieved Context:")
for chunk in result["context"]:
    print(f"- Rank {chunk['rank']}: {chunk['text'][:100]}...")
```

### `build_prompt(question, context_chunks) -> str`

Build a prompt for the LLM using retrieved context.

**Parameters:**

- `question` (str): User's question
- `context_chunks` (List[Dict]): Retrieved relevant chunks

**Returns:**

- `str`: Formatted prompt string

**Example:**

```python
# Manual prompt building
query_embedding = embedder.embed_text("What is AI?")
raw_results = store.search(query_embedding, n_results=3)
context_chunks = rag._format_results(raw_results)

prompt = rag.build_prompt("What is AI?", context_chunks)
print(prompt)
```

### `build_prompt_with_chat_history(question, context_chunks, chat_history=None) -> str`

Build a prompt that includes chat history for follow-up questions.

**Parameters:**

- `question` (str): Current question
- `context_chunks` (List[Dict]): Retrieved chunks
- `chat_history` (Optional[List[Dict]]): List of `{"role": "user/assistant", "content": "..."}`

**Returns:**

- `str`: Formatted prompt with history

**Example:**

```python
chat_history = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "How does it work?"}
]

query_embedding = embedder.embed_text("How does it work?")
raw_results = store.search(query_embedding, n_results=3)
context_chunks = rag._format_results(raw_results)

prompt = rag.build_prompt_with_chat_history(
    "How does it work?",
    context_chunks,
    chat_history
)
```

### `interactive_qa_loop()`

Interactive Q&A loop for command-line usage.

**Example:**

```python
# Start interactive mode
rag.interactive_qa_loop()

# User can then:
# - Ask questions
# - Type 'context' to toggle context display
# - Type 'quit' to exit
```

## Complete Example

```python
from src import (
    PDFParser,
    TextChunker,
    EmbeddingModel,
    VectorStore,
    RAGInterface
)
from src.llm_providers import OpenAILLM

# 1. Extract and chunk PDF
parser = PDFParser()
text = parser.extract_text("document.pdf")

chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_text(text)

# 2. Create embeddings
embedder = EmbeddingModel()
embeddings = embedder.embed_batch(chunks)

# 3. Store in vector database
store = VectorStore(
    collection_name="my_docs",
    persist_directory="./chroma_db"
)
store.add_chunks(chunks, embeddings)

# 4. Initialize RAG
llm = OpenAILLM(model="gpt-3.5-turbo")
rag = RAGInterface(
    embedding_model=embedder,
    vector_store=store,
    llm=llm,
    n_results=3
)

# 5. Ask questions
questions = [
    "What is the main topic?",
    "Who are the authors?",
    "What are the key findings?"
]

for question in questions:
    result = rag.answer_question(question, show_context=True)

    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")

    if result.get('context'):
        print(f"Based on {len(result['context'])} chunks")
```

## Customizing Prompts

### Using Custom Prompt Templates

```python
from src.rag import PromptTemplates

# Extractive QA (exact quotes)
prompt = PromptTemplates.extractive_qa(
    question="What is the definition?",
    context="Machine learning is..."
)

# Summarization
prompt = PromptTemplates.summarization(
    question="Summarize the key points",
    context="..."
)

# Comparative analysis
prompt = PromptTemplates.comparative(
    question="Compare X and Y",
    context="..."
)

# Use with LLM
answer = llm.generate(prompt)
```

### Custom Prompt Format

```python
class CustomRAG(RAGInterface):
    def build_prompt(self, question, context_chunks):
        context_texts = [c['text'] for c in context_chunks]
        combined = "\n\n".join(context_texts)

        return f"""<|system|>
You are a helpful assistant.
</s>
<|context|>
{combined}
</s>
<|user|>
{question}
</s>
<|assistant|>"""

# Use custom RAG
custom_rag = CustomRAG(embedder, store, llm)
```

## Advanced Usage

### Multi-Step Reasoning

```python
def multi_step_qa(rag, question):
    """Break down complex questions into steps."""

    # Step 1: Identify sub-questions
    result1 = rag.answer_question(
        f"Break down this question into sub-questions: {question}",
        show_context=False
    )

    # Step 2: Answer each sub-question
    sub_questions = result1["answer"].split("\n")
    sub_answers = []

    for sub_q in sub_questions:
        if sub_q.strip():
            result = rag.answer_question(sub_q.strip())
            sub_answers.append(result["answer"])

    # Step 3: Synthesize final answer
    synthesis_prompt = f"""Based on these answers:
{chr(10).join(sub_answers)}

Provide a comprehensive answer to: {question}"""

    final_result = rag.answer_question(synthesis_prompt)
    return final_result["answer"]
```

### Confidence Scoring

```python
def answer_with_confidence(rag, question):
    """Get answer with confidence score."""
    result = rag.answer_question(question, show_context=True)

    # Check distance of retrieved chunks
    distances = [c.get('distance', 1.0) for c in result['context']]
    avg_distance = sum(distances) / len(distances)

    # Lower distance = higher confidence
    confidence = 1.0 - min(avg_distance, 1.0)

    return {
        "answer": result["answer"],
        "confidence": confidence,
        "context_quality": "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
    }

result = answer_with_confidence(rag, "What is AI?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f} ({result['context_quality']})")
```

### Streaming Responses

```python
# For streaming LLM responses (if supported by provider)
class StreamingRAG(RAGInterface):
    def answer_question_stream(self, question):
        """Stream answer token by token."""
        # Retrieve context
        query_embedding = self.embedding_model.embed_text(question)
        raw_results = self.vector_store.search(query_embedding, self.n_results)
        context_chunks = self._format_results(raw_results)

        # Build prompt
        prompt = self.build_prompt(question, context_chunks)

        # Stream response (if LLM supports it)
        if hasattr(self.llm, 'stream'):
            for chunk in self.llm.stream(prompt):
                yield chunk
        else:
            # Fallback to regular generation
            yield self.llm.generate(prompt)

# Usage
streaming_rag = StreamingRAG(embedder, store, llm)
for chunk in streaming_rag.answer_question_stream("What is AI?"):
    print(chunk, end='', flush=True)
```

## Tuning Parameters

### Number of Results (n_results)

```python
# More context (slower, more comprehensive)
rag = RAGInterface(embedder, store, llm, n_results=5)

# Less context (faster, more focused)
rag = RAGInterface(embedder, store, llm, n_results=2)

# Default (balanced)
rag = RAGInterface(embedder, store, llm, n_results=3)
```

### Temperature

```python
# Deterministic (factual)
result = rag.answer_question(question, temperature=0.0)

# Balanced
result = rag.answer_question(question, temperature=0.7)

# Creative
result = rag.answer_question(question, temperature=1.0)
```

### Max Tokens

```python
# Short answer
result = rag.answer_question(question, max_tokens=100)

# Detailed answer
result = rag.answer_question(question, max_tokens=1000)
```

## Error Handling

```python
try:
    result = rag.answer_question("What is AI?")
    print(result["answer"])
except Exception as e:
    print(f"Error: {e}")

    # Fallback
    print("Could not generate answer. Returning search results only:")
    query_emb = embedder.embed_text("What is AI?")
    results = store.search(query_emb, n_results=3)
    for doc in results['documents'][0]:
        print(f"- {doc[:200]}...")
```

## Performance Tips

- **Reduce n_results** for faster responses
- **Use temperature=0.0-0.3** for factual answers
- **Enable show_context** only for debugging
- **Batch questions** when possible

## Dependencies

- All core modules (extract, chunk, embed, vector_store)
- LLM providers module

## See Also

- [LLM Providers API](llm-providers.md) - LLM configuration
- [VectorStore API](vector-store.md) - Vector search
- [Getting Started](../getting-started/quickstart.md) - Quick start guide
