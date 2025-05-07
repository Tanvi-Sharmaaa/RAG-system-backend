import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model_name = "intfloat/multilingual-e5-large-instruct"
model = SentenceTransformer(embedding_model_name)


# Sample Tickets 
tickets = [
    {
        "title": "Login failure on Safari for SSO users",
        "browser": "Safari 16.3",
        "os": "macOS Ventura",
        "customer_type": "Enterprise",
        "issue": "Redirect loop during SSO login",
        "resolution": "Clear cookies, update Safari settings to allow cross-site tracking."
    },
    {
        "title": "Generic login issues",
        "browser": "All",
        "customer_type": "Mixed",
        "issue": "Password reset email not received",
        "resolution": "Whitelist support domain in email settings."
    },
    {
        "title": "Login error specific to Chrome extensions",
        "browser": "Chrome",
        "customer_type": "SMB",
        "issue": "Conflict with password manager extension",
        "resolution": "Disable conflicting extension."
    }
]

# Format ticket content for embedding
ticket_texts = [
    f"{t['title']}. Browser: {t['browser']}. OS: {t.get('os', 'Unknown')}. "
    f"Customer: {t['customer_type']}. Issue: {t['issue']}. Resolution: {t['resolution']}"
    for t in tickets
]

# Embedding and Indexing
ticket_embeddings = model.encode(ticket_texts, normalize_embeddings=True)
dimension = ticket_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(ticket_embeddings)


# Streamlit UI 
st.title("Semantic Support Ticket Retriever (RAG Prototype)")
st.markdown("Get contextual answers based on historic support tickets.")

query = st.text_input("Enter your support query:")

if query:
    # Embed the query
    query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")

    # Retrieve top-k similar tickets
    k = 2
    distances, indices = index.search(query_embedding, k)

    # Display retrieved tickets
    st.subheader(" Top Matching Tickets")
    retrieved_tickets = []
    for idx in indices[0]:
        ticket = ticket_texts[idx]
        retrieved_tickets.append(ticket)
        st.markdown(f"**Ticket {idx + 1}:** {ticket}")


    # Feedback Section
    st.subheader("Was this result helpful?")
    col1, col2 = st.columns(2)

    # Create a rating system
    rating = st.slider("Rate the relevance of this result (1 = not helpful, 5 = very helpful)", 1, 5, 3)

    # Text area for additional comments
    comments = st.text_area("Provide any additional comments (optional)", "")

    # Category selection for feedback
    category = st.selectbox("How would you categorize this result?", 
                            ["Relevant", "Too Generic", "Missing Context", "Irrelevant", "Other"])

    # Create a button to submit feedback
    with col1:
        if st.button("Submit Feedback"):
            # Process the feedback (you can implement saving to a database/file here)
            st.success(f"Thank you for your feedback! You rated this result: {rating}/5.")
            if comments:
                st.write(f"Your comments: {comments}")
            st.write(f"Category of feedback: {category}")

    # If no feedback 
    with col2:
        if st.button("No, I need different results"):
            st.warning("We'll use this feedback to improve the search results.")


# Footer
st.markdown("---")
st.markdown(
    "**Model Used:** `intfloat/multilingual-e5-large-instruct` for semantic search | "
    "\n**Vector Store:** FAISS | **Retrieval Strategy:** Top-k (k=2)"
)
