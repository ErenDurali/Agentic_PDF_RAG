import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from collections import defaultdict
from langchain.schema.document import Document
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# API Key setup
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"  # Replace with your actual API key
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

from get_embedding_function import get_embedding_function

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(current_dir, "chroma")

# Token usage tracking
token_usage = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
}

def track_token_usage(usage_data):
    """Track token usage from API responses"""
    if usage_data and hasattr(usage_data, 'prompt_tokens'):
        token_usage["prompt_tokens"] += usage_data.prompt_tokens
        token_usage["completion_tokens"] += usage_data.completion_tokens
        token_usage["total_tokens"] += usage_data.total_tokens
    return token_usage

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

CLASSIFICATION_PROMPT = """
You are an AI assistant helping determine if a query requires consulting specific documents.

Query: {query}

Does this query require consulting documents about course materials, or can it be answered using general knowledge?
Respond with either:
1. "RAG" - if the query needs to consult documents (e.g. specific questions about course content)
2. "General" - if the query can be answered with general knowledge without consulting documents

Explanation:
"""

SUMMARIZATION_PROMPT = """
Create a summary based on the following content from the document:

{context}

Type of summary requested: {summary_type}

Your task is to create a coherent, well-structured {summary_type} that captures the essential information from the provided content.
"""

# Define which exceptions should be retried
def is_rate_limit_or_timeout(exception):
    error_str = str(exception).lower()
    return ("429" in error_str or 
            "quota" in error_str or
            "exhausted" in error_str or
            "timeout" in error_str or 
            "too many requests" in error_str or
            "temporarily" in error_str)

# Apply this retry decorator to functions that call the API
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),  # Try 3 times
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Wait between attempts
    reraise=True
)
def query_with_retry(model, prompt):
    """Call the model with retry logic for better reliability."""
    try:
        response = model.invoke(prompt)
        # Track token usage if available
        if hasattr(response, 'usage'):
            track_token_usage(response.usage)
        return response
    except Exception as e:
        if is_rate_limit_or_timeout(e):
            # This will be caught by the retry decorator
            raise e
        else:
            # Other errors are re-raised immediately
            raise

def classify_query(query_text: str):
    """Determine if the query needs RAG or can be answered with general knowledge."""
    prompt_template = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
    prompt = prompt_template.format(query=query_text)
    
    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    
    try:
        response = query_with_retry(model, prompt)
        
        # Extract response text
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Check if the response indicates RAG is needed
        if "RAG" in response_text.split("\n")[0]:
            return True
        return False
    except Exception as e:
        # On failure, default to using RAG to be safe
        return True

def query_direct(query_text: str):
    """Answer the query using the model's general knowledge."""
    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    try:
        response = query_with_retry(model, query_text)
        # Extract just the text content from the response object
        response_text = response.content if hasattr(response, 'content') else str(response)
        return response_text
    except Exception as e:
        # If all retries fail, return an error message
        return f"Unable to process your request due to API limitations. Error: {str(e)}"

def assess_retrieval_quality(query_text: str, context_text: str):
    """Assess if the retrieved documents are relevant to the query."""
    # Shorten context if it's too long to save tokens
    if len(context_text) > 10000:
        context_text = context_text[:5000] + "...[content truncated]..." + context_text[-5000:]
    
    assessment_prompt = f"""
    Task: Evaluate the relevance of retrieved context to a query.
    
    Query: {query_text}
    
    Retrieved Context:
    {context_text}
    
    Is the retrieved context relevant and sufficient to answer the query?
    Rate on a scale of 0-10, where:
    - 0-3: Irrelevant or insufficient
    - 4-6: Partially relevant
    - 7-10: Highly relevant and sufficient
    
    Provide only the numerical score:
    """
    
    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    
    try:
        response = query_with_retry(model, assessment_prompt)
        
        # Extract just the text content from the response object
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract the numerical score
        score = 0
        try:
            # Try to extract just the number
            score = int(response_text.strip().split()[0])
        except:
            # If that fails, default to 5
            score = 5
        
        return score
    except Exception as e:
        # In case of API errors, default to 5 (middle score)
        return 5

def query_rag_with_confidence(query_text: str):
    """Answer query with RAG and return confidence score."""
    # Add a brief delay to help with rate limiting
    time.sleep(0.5)
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Assess retrieval quality
    quality_score = assess_retrieval_quality(query_text, context_text)
    
    # If quality is too low, fall back to general knowledge
    if quality_score < 4:
        response_text = query_direct(query_text)
        return response_text, ["No relevant documents found - used general knowledge"], quality_score
    
    # Otherwise, proceed with RAG
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    
    try:
        response = query_with_retry(model, prompt)
        
        # Extract just the text content from the response object
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Group sources by PDF file
        source_dict = defaultdict(list)
        for doc, _score in results:
            source_id = doc.metadata.get("id", "Unknown")
            if source_id and ":" in source_id:
                pdf_name = source_id.split(":")[0].split("/")[-1]  # Extract PDF filename
                page_chunk = ":".join(source_id.split(":")[1:])    # Extract page and chunk info
                source_dict[pdf_name].append(page_chunk)
        
        # Format sources as "pdf_name (page:chunk, page:chunk, ...)"
        formatted_sources = []
        for pdf, locations in source_dict.items():
            formatted_sources.append(f"{pdf} ({', '.join(locations)})")
        
        return response_text, formatted_sources, quality_score
    except Exception as e:
        # If all retries fail, fall back to direct query
        response_text = query_direct(query_text)
        return response_text, ["Error retrieving from documents - used general knowledge"], 3

def generate_summary(document_name: str, summary_type: str = "brief overview"):
    """Generate a summary of a specific document."""
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Create a search query focused on the document
    search_query = f"content from {document_name}"
    
    # Search the DB for chunks from this document - increase k to get more chunks
    results = db.similarity_search_with_score(search_query, k=20)
    
    # Filter results to only include chunks from the specified document
    document_results = []
    for doc, score in results:
        doc_id = doc.metadata.get("id", "")
        if document_name.lower() in doc_id.lower():
            document_results.append((doc, score))
    
    # If no chunks found for this document, return an error message
    if not document_results:
        return f"No content found for document '{document_name}'.", [], 0
    
    # Sort by page and chunk number to maintain document order
    document_results.sort(key=lambda x: (
        # Sort by source and page number embedded in the id
        x[0].metadata.get("source", ""),
        int(x[0].metadata.get("page", "0"))
    ))
    
    # Process the document in batches if it's very large
    max_chunks_per_batch = 10
    batched_summaries = []
    
    for i in range(0, len(document_results), max_chunks_per_batch):
        batch = document_results[i:i + max_chunks_per_batch]
        
        # Extract content from these chunks
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in batch])
        
        # Create summarization prompt for this batch
        batch_prompt = f"""
        Create a summary of the following section of the document:
        
        {context_text}
        
        Focus on the key information and main points from this section.
        """
        
        # Generate summary for this batch
        model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
        
        try:
            response = query_with_retry(model, batch_prompt)
            
            # Extract response text
            response_text = response.content if hasattr(response, 'content') else str(response)
            batched_summaries.append(response_text)
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            # On error, continue with what we have so far
            batched_summaries.append(f"Error summarizing section: {str(e)}")
    
    # Now create a final combined summary from all batch summaries
    combined_summary = "\n\n".join(batched_summaries)
    
    final_prompt_template = ChatPromptTemplate.from_template("""
    Below are summaries of different sections from the document '{document_name}':
    
    {combined_summary}
    
    Create a coherent {summary_type} that combines all this information.
    If the summary type is 'brief overview', keep it concise (1-2 paragraphs).
    If it's a 'detailed summary', provide comprehensive coverage of all important content.
    If it's 'key points & concepts', organize the main ideas and concepts in a structured way.
    """)
    
    final_prompt = final_prompt_template.format(
        document_name=document_name,
        combined_summary=combined_summary,
        summary_type=summary_type
    )
    
    # Generate final summary
    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    
    try:
        final_response = query_with_retry(model, final_prompt)
        
        # Extract response text
        final_response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
    except Exception as e:
        # If final summary fails, return what we have so far
        final_response_text = f"Error generating final summary. Here are the section summaries:\n\n{combined_summary}"
    
    # Group sources by PDF file
    source_dict = defaultdict(list)
    for doc, _ in document_results:
        source_id = doc.metadata.get("id", "Unknown")
        if source_id and ":" in source_id:
            pdf_name = source_id.split(":")[0].split("/")[-1]  # Extract PDF filename
            page_chunk = ":".join(source_id.split(":")[1:])    # Extract page and chunk info
            source_dict[pdf_name].append(page_chunk)
    
    # Format sources as "pdf_name (page:chunk, page:chunk, ...)"
    formatted_sources = []
    for pdf, locations in source_dict.items():
        formatted_sources.append(f"{pdf} ({', '.join(locations)})")
    
    # For summaries, quality is based on the breadth of document coverage
    quality_score = min(10, len(document_results) // 2)  # Higher score for more document chunks
    
    return final_response_text, formatted_sources, quality_score

def generate_comprehensive_summary(document_name: str, summary_type: str = "brief overview"):
    """Generate a more comprehensive summary by processing the entire document."""
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Retrieve ALL chunks from this document
    # First get all documents in the database
    all_docs = db.get()
    
    # Filter to only those from our target document
    document_chunks = []
    if all_docs and 'metadatas' in all_docs and all_docs['metadatas']:
        for i, metadata in enumerate(all_docs['metadatas']):
            if metadata and 'id' in metadata:
                doc_id = metadata['id']
                if document_name.lower() in doc_id.lower():
                    # Create a document object with content and metadata
                    doc = Document(
                        page_content=all_docs['documents'][i],
                        metadata=metadata
                    )
                    document_chunks.append((doc, 1.0))  # Add a dummy score of 1.0
    
    # If no chunks found for this document, return an error message
    if not document_chunks:
        return f"No content found for document '{document_name}'.", [], 0
    
    # Sort by page and chunk to maintain document order
    document_chunks.sort(key=lambda x: (
        # Extract page number from metadata
        int(x[0].metadata.get("page", "0")),
        # Extract chunk index from id (after the last colon)
        int(x[0].metadata["id"].split(":")[-1]) if ":" in x[0].metadata.get("id", "") else 0
    ))
    
    print(f"Processing {len(document_chunks)} chunks from document: {document_name}")
    
    # Process in batches of ~8000 tokens (roughly 10-15 chunks depending on size)
    batch_size = 10
    batched_summaries = []
    
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i + batch_size]
        
        # Get page range for this batch for reference
        start_page = batch[0][0].metadata.get("page", "?")
        end_page = batch[-1][0].metadata.get("page", "?")
        
        # Extract content from these chunks
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in batch])
        
        # Create summarization prompt for this batch
        batch_prompt = f"""
        Summarize the following section from document '{document_name}' (pages {start_page}-{end_page}):
        
        {context_text}
        
        Extract the main points, key information, and important details from this section.
        """
        
        # Generate summary for this batch
        model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
        
        try:
            response = query_with_retry(model, batch_prompt)
            
            # Extract response text
            response_text = response.content if hasattr(response, 'content') else str(response)
            batched_summaries.append(f"Pages {start_page}-{end_page}: {response_text}")
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
        except Exception as e:
            # On error, continue with what we have so far
            batched_summaries.append(f"Error summarizing pages {start_page}-{end_page}: {str(e)}")
    
    # Now create a final combined summary from all batch summaries
    combined_sections = "\n\n" + "\n\n".join(batched_summaries)
    
    # Create a prompt for bullet-point style summary
    bullet_point_format = """
    Create a well-structured bullet-point summary that:
    • Starts with a brief overview paragraph (not bulleted)
    • Uses a clear hierarchical structure with:
      • Main sections as primary bullet points with headings
      • Key concepts as secondary bullet points
      • Supporting details as tertiary bullet points
    • Groups related concepts together logically
    • Highlights definitions, principles, and key conclusions
    
    IMPORTANT FORMATTING INSTRUCTIONS:
    • Use circular bullets (•) for all bullet points (not asterisks or hyphens)
    • Use proper indentation for hierarchy:
      • Main points: No indentation
      • Sub-points: 2 spaces before the bullet
      • Sub-sub-points: 4 spaces before the bullet
    • Each bullet point should be on its own line
    • Add an extra blank line after each major section for readability
    """
    
    final_prompt = f"""
    Below are summaries of different sections from the document '{document_name}':
    
    {combined_sections}
    
    {bullet_point_format}
    
    Make the summary highly readable with good organization and flow.
    """
    
    # Generate final summary
    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    
    try:
        final_response = query_with_retry(model, final_prompt)
        
        # Extract response text
        final_response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
    except Exception as e:
        # If final summary fails, return what we have so far
        final_response_text = f"Error generating final summary. Here are the section summaries:\n\n{combined_sections}"
    
    # Format sources info
    formatted_sources = [f"{document_name} (Full document analysis, {len(document_chunks)} sections)"]
    
    # High confidence since we processed the entire document
    quality_score = 10
    
    return final_response_text, formatted_sources, quality_score

def get_token_usage():
    """Return the current token usage statistics."""
    return token_usage

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--summary", action="store_true", help="Generate a summary instead of answering a question.")
    parser.add_argument("--summary_type", type=str, default="brief overview", help="Type of summary to generate.")
    args = parser.parse_args()
    query_text = args.query_text
    
    start_time = time.time()
    
    if args.summary:
        print(f"Generating {args.summary_type} summary...")
        response, sources, confidence = generate_summary(query_text, args.summary_type)
        print(f"Summary: {response}")
        print(f"Sources: {sources}")
        print(f"Confidence: {confidence}/10")
    else:
        # Determine if RAG is needed
        needs_rag = classify_query(query_text)
        
        if needs_rag:
            print("Using RAG to answer the query...")
            response, sources, confidence = query_rag_with_confidence(query_text)
            print(f"Response: {response}")
            print(f"Sources: {sources}")
            print(f"Confidence: {confidence}/10")
        else:
            print("Answering with general knowledge...")
            response = query_direct(query_text)
            print(f"Response: {response}")
            print(f"Sources: General knowledge")
    
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    
    # Print token usage
    usage = get_token_usage()
    print(f"\nToken Usage:")
    print(f"  Prompt tokens: {usage['prompt_tokens']}")
    print(f"  Completion tokens: {usage['completion_tokens']}")
    print(f"  Total tokens: {usage['total_tokens']}")

if __name__ == "__main__":
    main()