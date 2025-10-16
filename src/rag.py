"""
RAG (Retrieval-Augmented Generation) Interface
Combines document retrieval with LLM generation to answer questions.
This module is MODEL-AGNOSTIC - works with any LLM!
"""

from typing import List, Dict, Optional
from .llm_providers import BaseLLM, get_available_llm
from .embed import EmbeddingModel
from .vector_store import VectorStore


class RAGInterface:
    """
    RAG system that retrieves relevant context and generates answers.
    The prompt construction here works with ANY LLM!
    """
    
    def __init__(
        self, 
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        llm: Optional[BaseLLM] = None,
        n_results: int = 3
    ):
        """
        Initialize RAG interface.
        
        Args:
            embedding_model: Model for creating embeddings
            vector_store: Vector database with document chunks
            llm: LLM provider (auto-detects if None)
            n_results: Number of chunks to retrieve for context
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm or get_available_llm()
        self.n_results = n_results
    
    def build_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Build a prompt for the LLM using retrieved context.
        
        THIS IS MODEL-AGNOSTIC! Works with any LLM.
        The structure follows best practices for RAG systems.
        
        Args:
            question: User's question
            context_chunks: Retrieved relevant chunks from vector DB
            
        Returns:
            Formatted prompt string
        """
        # Extract just the text from chunks
        context_texts = [chunk['text'] for chunk in context_chunks]
        
        # Combine all context with separators
        combined_context = "\n\n---\n\n".join([
            f"[Context {i+1}]:\n{text}" 
            for i, text in enumerate(context_texts)
        ])
        
        # Build the complete prompt
        # This structure works well with most LLMs (GPT, Claude, Llama, etc.)
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

CONTEXT FROM DOCUMENT:
{combined_context}

---

USER QUESTION:
{question}

---

INSTRUCTIONS:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information, say so clearly
- Be specific and cite relevant details from the context
- Keep your answer concise but complete
- If there are multiple relevant points, organize them clearly

ANSWER:"""
        
        return prompt
    
    def build_prompt_with_chat_history(
        self, 
        question: str, 
        context_chunks: List[Dict],
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Build a prompt that includes chat history for follow-up questions.
        
        Args:
            question: Current question
            context_chunks: Retrieved chunks
            chat_history: List of {"role": "user/assistant", "content": "..."}
            
        Returns:
            Formatted prompt with history
        """
        context_texts = [chunk['text'] for chunk in context_chunks]
        combined_context = "\n\n---\n\n".join([
            f"[Context {i+1}]:\n{text}" 
            for i, text in enumerate(context_texts)
        ])
        
        # Build chat history section
        history_section = ""
        if chat_history:
            history_section = "\nPREVIOUS CONVERSATION:\n"
            for msg in chat_history[-5:]:  # Last 5 messages
                role = msg['role'].upper()
                content = msg['content']
                history_section += f"{role}: {content}\n"
            history_section += "\n---\n"
        
        prompt = f"""You are a helpful AI assistant answering questions about a document.

CONTEXT FROM DOCUMENT:
{combined_context}

---
{history_section}
CURRENT QUESTION:
{question}

---

INSTRUCTIONS:
- Answer based on the provided context and conversation history
- Maintain coherence with previous responses
- If asking about "it" or "that", refer to conversation history for context
- Be specific and cite details from the document context

ANSWER:"""
        
        return prompt
    
    def answer_question(
        self, 
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        show_context: bool = False
    ) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            show_context: Whether to return retrieved chunks
            
        Returns:
            Dictionary with answer and optionally context
        """
        # Step 1: Retrieve relevant chunks
        query_embedding = self.embedding_model.embed_text(question)
        raw_results = self.vector_store.search(query_embedding, n_results=self.n_results)
        
        # Format results
        context_chunks = self._format_results(raw_results)
        
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information in the document to answer your question.",
                "context": [] if show_context else None
            }
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, context_chunks)
        
        # Step 3: Generate answer
        answer = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Return result
        result = {
            "answer": answer.strip(),
            "question": question
        }
        
        if show_context:
            result["context"] = context_chunks
            result["prompt"] = prompt  # Useful for debugging
        
        return result
    
    def _format_results(self, raw_results: Dict) -> List[Dict]:
        """Format raw vector search results."""
        formatted = []
        
        documents = raw_results.get('documents', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        distances = raw_results.get('distances', [[]])[0]
        
        for i, doc in enumerate(documents):
            result = {
                'text': doc,
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'distance': distances[i] if i < len(distances) else None,
                'rank': i + 1
            }
            formatted.append(result)
        
        return formatted
    
    def interactive_qa_loop(self):
        """
        Interactive Q&A loop with LLM-powered answers.
        """
        print("\n" + "="*80)
        print("RAG-Powered Q&A System")
        print("="*80)
        print(f"Using LLM: {self.llm.__class__.__name__}")
        print("Ask questions about your document. Type 'quit' to exit.")
        print("Type 'context' to toggle showing retrieved context.")
        print("="*80 + "\n")
        
        show_context = False
        
        while True:
            try:
                # Get user input
                question = input("❓ Question: ").strip()
                
                # Check for commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'context':
                    show_context = not show_context
                    status = "ON" if show_context else "OFF"
                    print(f"📋 Context display: {status}\n")
                    continue
                
                if not question:
                    continue
                
                # Get answer
                print("\n🤔 Thinking...")
                result = self.answer_question(
                    question=question,
                    temperature=0.7,
                    show_context=show_context
                )
                
                # Display answer
                print("\n" + "="*80)
                print("💡 ANSWER:")
                print("="*80)
                print(result['answer'])
                print("="*80)
                
                # Optionally show context
                if show_context and result.get('context'):
                    print("\n📚 RETRIEVED CONTEXT:")
                    print("-"*80)
                    for chunk in result['context']:
                        print(f"\n[Chunk {chunk['rank']}]:")
                        print(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                    print("="*80)
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


# Different prompt templates for different use cases
class PromptTemplates:
    """Collection of prompt templates for various scenarios."""
    
    @staticmethod
    def basic_qa(question: str, context: str) -> str:
        """Simple Q&A prompt."""
        return f"""Context: {context}

Question: {question}

Answer:"""
    
    @staticmethod
    def extractive_qa(question: str, context: str) -> str:
        """Prompt for extracting exact quotes."""
        return f"""Based on the following context, answer the question using EXACT quotes.

Context:
{context}

Question: {question}

Instructions: Quote directly from the context. If no exact answer exists, say "Not found in context."

Answer:"""
    
    @staticmethod
    def summarization(question: str, context: str) -> str:
        """Prompt for summarizing information."""
        return f"""Summarize the following information to answer the question.

Content:
{context}

Question: {question}

Summary:"""
    
    @staticmethod
    def comparative(question: str, context: str) -> str:
        """Prompt for comparing information."""
        return f"""Compare and analyze the information below to answer the question.

Information:
{context}

Question: {question}

Analysis:"""


if __name__ == "__main__":
    print("RAG Interface Module")
    print("\nThis module provides:")
    print("1. Model-agnostic prompt construction")
    print("2. Retrieval + Generation pipeline")
    print("3. Multiple prompt templates")
    print("\nUse via main_rag.py for full functionality")

