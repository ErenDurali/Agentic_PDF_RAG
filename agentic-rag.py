import streamlit as st
import os
import sys
import subprocess
from datetime import datetime

# Ensure the script can find modules in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from query_data import (query_rag_with_confidence, query_direct, classify_query, 
                      generate_comprehensive_summary)

# Constants
DATA_PATH = os.path.join(current_dir, "data")
CHROMA_PATH = os.path.join(current_dir, "chroma")

def handle_api_error(e):
    """Handle common API errors with user-friendly messages."""
    error_str = str(e).lower()
    
    if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
        st.error("😓 API quota limit reached. Please try again later or check your API key limits.")
        st.info("💡 Tips: You can try reducing the number of requests or upgrading your API plan.")
    elif "403" in error_str or "permission" in error_str:
        st.error("🔒 API permission error. Please check your API key and permissions.")
    else:
        st.error(f"An error occurred: {str(e)}")
        st.error("Make sure your database is properly set up with PDF documents.")
        st.error(f"Error details: {e}")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Agentic RAG Assistant")
st.write("Your intelligent study companion - ask questions about your course materials and get personalized assistance.")

# Create sidebar
with st.sidebar:
    st.header("Settings & Tools")
    
    # App settings
    st.subheader("App Settings")
    show_sources = st.checkbox("Show sources", value=False, help="Toggle to show or hide document sources")
    
    # Database management section
    st.subheader("Database Management")
    
    # Show paths
    st.info(f"📁 Data folder: {DATA_PATH}")
    st.info(f"🗄️ Database: {CHROMA_PATH}")
    
    # Check for new PDFs and update database
    if st.button("Check for new PDFs"):
        with st.spinner("Checking for new content..."):
            try:
                # Run the populate_database script
                result = subprocess.run(
                    ["python", os.path.join(current_dir, "populate_database.py")],
                    capture_output=True,
                    text=True
                )
                st.success("Database updated successfully!")
                st.code(result.stdout)
            except Exception as e:
                st.error(f"Error updating database: {e}")
    
    # Option to reset database
    if st.button("Reset database", type="secondary"):
        with st.spinner("Resetting database..."):
            try:
                # Run the populate_database script with reset flag
                result = subprocess.run(
                    ["python", os.path.join(current_dir, "populate_database.py"), "--reset"],
                    capture_output=True,
                    text=True
                )
                st.success("Database reset successfully!")
                st.code(result.stdout)
            except Exception as e:
                st.error(f"Error resetting database: {e}")

# Define function to get confidence badge
def get_confidence_badge(confidence_level):
    """Return a colored badge based on confidence level."""
    if confidence_level >= 8:
        return "🟢 High Confidence"
    elif confidence_level >= 5:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"

# Create tabs for different functions
tab1, tab2, tab3 = st.tabs(["Ask Questions", "Summarize Content", "Explain Concepts"])

with tab1:
    st.header("Ask Questions")
    query = st.text_input("Ask a question about your course materials:", key="query")
    
    if st.button("Submit", key="submit_query") or query:
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing your question..."):
                try:
                    # Determine if RAG is needed
                    needs_rag = classify_query(query)
                    
                    if needs_rag:
                        st.write("🔍 Using course materials to find information...")
                        response, sources, confidence = query_rag_with_confidence(query)
                        
                        # Display confidence
                        st.write(get_confidence_badge(confidence))
                        
                        # Display response
                        st.subheader("Answer")
                        st.write(response)
                        
                        # Display sources only if enabled
                        if show_sources:
                            st.subheader("Sources")
                            if any(source.startswith("No relevant") for source in sources):
                                st.warning("No highly relevant content found - response generated using general knowledge")
                            else:
                                for source in sources:
                                    st.write(f"- {source}")
                    else:
                        st.write("💡 Answering with general knowledge...")
                        response = query_direct(query)
                        
                        # Display response
                        st.subheader("Answer")
                        st.write(response)
                        
                        # Note about no sources only if sources are enabled
                        if show_sources:
                            st.info("This answer was generated using general knowledge without consulting your course materials.")
                        
                except Exception as e:
                    handle_api_error(e)

with tab2:
    st.header("Summarize Content")
    
    # Allow user to choose a document to summarize
    pdf_files = []
    if os.path.exists(DATA_PATH):
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.warning("No PDF files found in the data folder. Please add some PDFs first.")
    else:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            selected_pdf = st.selectbox("Select a document to summarize", pdf_files)
            
            summary_type = st.radio(
                "Choose summary type:",
                ["Brief overview", 
                 "Comprehensive summary",
                 "Detailed concept analysis"]
            )
        
        with col2:
            processing_mode = st.radio(
                "Processing mode:",
                ["Standard", "Comprehensive (for long documents)", "Conceptual focus"],
                help="Select the processing mode based on your needs"
            )
            
            # Add note that all summaries will use bullet points
            st.info("All summaries will be presented in a structured bullet-point format for better readability.")
            
            if processing_mode == "Conceptual focus":
                st.info("Conceptual focus mode emphasizes understanding over mathematical details.")
        
        if st.button("Generate Summary", key="generate_summary"):
            with st.spinner("Generating summary... This may take a minute for long documents"):
                try:
                    # Build a query for summarization based on selection but always requesting bullet points
                    if summary_type == "Brief overview":
                        summary_format = """
                        Provide a concise overview of the document in bullet-point format:
                        • Begin with a brief introduction paragraph (not bulleted)
                        • Focus on the main ideas and concepts
                        • Keep the total length to about 5-7 main bullet points
                        • Use sub-bullets sparingly for key details
                        """
                    elif summary_type == "Comprehensive summary":
                        summary_format = """
                        Create a comprehensive bullet-point summary that:
                        • Begins with a brief overview paragraph
                        • Covers all major sections and concepts
                        • Uses a clear hierarchical structure with main points and supporting details
                        • Provides sufficient depth while maintaining readability
                        • Groups related concepts together logically
                        """
                    else:  # Detailed concept analysis
                        summary_format = """
                        Create a detailed analysis of the key concepts in bullet-point format:
                        • Start with a paragraph introducing the main focus of the document
                        • Organize concepts into logical categories with clear headings
                        • For each concept, include:
                          - A clear definition
                          - Its significance or application
                          - Relationship to other concepts
                        • Highlight theoretical principles and frameworks
                        • Emphasize conceptual understanding over mathematical details
                        """
                    
                    # Adjust for processing mode
                    if processing_mode == "Conceptual focus":
                        mode_adjustment = """
                        Focus on explaining the conceptual understanding rather than mathematical details.
                        When math concepts are mentioned, explain their meaning and purpose rather than
                        showing formulas. Emphasize the 'why' over the 'how'.
                        
                        IMPORTANT FORMATTING INSTRUCTIONS:
                        • Use circular bullets (•) for all bullet points (not asterisks, numbers, or hyphens)
                        • Use proper indentation for hierarchy:
                            • Main points: No indentation
                            • Sub-points: 2 spaces before the bullet
                            • Sub-sub-points: 4 spaces before the bullet
                        • Bold any section headings within the bullet points
                        • Each bullet point should be on its own line
                        • Add an extra blank line after each major section for readability
                        """
                    else:
                        mode_adjustment = """
                        IMPORTANT FORMATTING INSTRUCTIONS:
                        • Use circular bullets (•) for all bullet points (not asterisks, numbers, or hyphens)
                        • Use proper indentation for hierarchy:
                            • Main points: No indentation
                            • Sub-points: 2 spaces before the bullet
                            • Sub-sub-points: 4 spaces before the bullet
                        • Bold any section headings within the bullet points
                        • Each bullet point should be on its own line
                        • Add an extra blank line after each major section for readability
                        """
                    
                    # Combine format and adjustments
                    summary_query = f"For the document '{selected_pdf}': {summary_format} {mode_adjustment}"
                    
                    # Use different summary function based on processing mode
                    if processing_mode == "Comprehensive (for long documents)":
                        response, sources, confidence = generate_comprehensive_summary(selected_pdf, "bullet-point " + summary_type)
                    else:
                        response, sources, confidence = query_rag_with_confidence(summary_query)
                    
                    # Standardize bullet points in the response
                    response = response.replace("* ", "• ").replace("- ", "• ")
                    formatted_response = ""
                    for line in response.split('\n'):
                        # Keep indentation but standardize to circles
                        if line.strip().startswith('*') or line.strip().startswith('-') or line.strip().startswith('•'):
                            indent = len(line) - len(line.lstrip())
                            bullet_text = line.strip()[1:].strip()
                            formatted_response += ' ' * indent + '• ' + bullet_text + '\n\n'  # Add extra line after each bullet
                        else:
                            formatted_response += line + '\n'
                    response = formatted_response
                    
                    # Display confidence
                    st.write(get_confidence_badge(confidence))
                    
                    # Display response
                    st.subheader(f"Summary of {selected_pdf}")
                    st.markdown(response)  # Using markdown for better formatting of bullet points
                    
                    # Display sources only if enabled
                    if show_sources:
                        st.subheader("Source")
                        for source in sources:
                            st.write(f"- {source}")
                        
                except Exception as e:
                    handle_api_error(e)

with tab3:
    st.header("Explain Concepts")
    
    # Free-form concept input
    concept = st.text_input("Enter a concept or term you want explained:", key="concept")
    
    # Context options
    context_option = st.radio(
        "How would you like the explanation?",
        ["Based on your course materials", "General explanation", "Simple explanation (ELI5)"]
    )
    
    if st.button("Explain", key="explain_concept") and concept:
        with st.spinner(f"Explaining '{concept}'..."):
            try:
                # Adjust the query based on the context option
                if context_option == "Based on your course materials":
                    query_text = f"Explain the concept of '{concept}' based on the course materials"
                    needs_rag = True
                elif context_option == "Simple explanation (ELI5)":
                    query_text = f"Explain the concept of '{concept}' in very simple terms, as if explaining to a 5-year-old"
                    needs_rag = False
                else:
                    query_text = f"Explain the concept of '{concept}'"
                    needs_rag = classify_query(query_text)
                
                if needs_rag:
                    st.write("🔍 Using course materials for the explanation...")
                    response, sources, confidence = query_rag_with_confidence(query_text)
                    
                    # Display confidence
                    st.write(get_confidence_badge(confidence))
                    
                    # Display response
                    st.subheader(f"Explanation of '{concept}'")
                    st.write(response)
                    
                    # Display sources only if enabled
                    if show_sources:
                        st.subheader("Sources")
                        if any(source.startswith("No relevant") for source in sources):
                            st.warning("No relevant information found in course materials - using general knowledge")
                        else:
                            for source in sources:
                                st.write(f"- {source}")
                else:
                    st.write("💡 Providing general explanation...")
                    response = query_direct(query_text)
                    
                    # Display response
                    st.subheader(f"Explanation of '{concept}'")
                    st.write(response)
                    
                    # Note about no sources only if sources are enabled
                    if show_sources:
                        st.info("This explanation was generated using general knowledge.")
                    
            except Exception as e:
                handle_api_error(e)

# Display usage instructions
with st.expander("How to use this app"):
    st.write("""
    ### Getting Started
    1. Add your PDF course materials to the 'data' folder
    2. Click "Check for new PDFs" in the sidebar to update the database
    3. Use the tabs to access different features:
        - **Ask Questions**: Get answers about your course materials
        - **Summarize Content**: Generate summaries of specific documents
        - **Explain Concepts**: Get explanations of course concepts
    
    ### Tips for Best Results
    - Be specific in your questions
    - For concept explanations, use proper terminology
    - Use "Conceptual focus" mode for better understanding of technical concepts
    - If the system can't find relevant information in your documents, it will fall back to general knowledge
    
    ### App Settings
    - Toggle "Show sources" in the sidebar to display or hide document references
    - Sources show which documents and pages information was drawn from
    
    ### Confidence Indicators
    - 🟢 High Confidence: Strong evidence found in documents
    - 🟡 Medium Confidence: Some relevant information found
    - 🔴 Low Confidence: Limited relevant information
    """)

# Add a footer with timestamp
st.markdown("---")
st.markdown(f"Agentic RAG Assistant | Created with LangChain and Google Gemini | Last updated: {datetime.now().strftime('%Y-%m-%d')}")