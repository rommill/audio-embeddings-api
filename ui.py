import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Audio AI Search", page_icon="🎵", layout="wide")

st.title("🎵 DevOps Project: Audio AI Search Engine")
st.markdown("""
This interface interacts with the **YAMNet** deep learning model to generate and search audio embeddings.
Use the tools below to index new sounds or find similarities in the database.
""")

# Sidebar for System Status
with st.sidebar:
    st.header("System Status")
    st.info("Stack: FastAPI + TensorFlow + Streamlit")
    if st.button("Check API Health"):
        try:
            # We use 'api' as the hostname because they are in the same Docker network
            res = requests.get("http://api:8000/docs", timeout=5)
            st.success("Backend API is Online ✅")
        except:
            st.error("Backend API is Offline ❌")

# Main Interface: Search Section
st.header("🔍 Similarity Search")
uploaded_file = st.file_uploader("Upload a .wav file to find matches", type=['wav'], key="search_upload")

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Audio")
        st.audio(uploaded_file)
    
    if st.button("🚀 Search Similar Sounds"):
        with st.spinner('Neural Network is analyzing embeddings...'):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                response = requests.post("http://api:8000/search", files=files)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    with col2:
                        st.subheader("Search Results")
                        if not results:
                            st.warning("No matches found. Please index some files first.")
                        else:
                            df = pd.DataFrame(results)
                            # Renaming columns for better UX
                            df.columns = ['Filename', 'Cosine Similarity']
                            st.table(df)
                            
                            top_match = results[0]
                            st.success(f"Best Match: **{top_match['filename']}**")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# Indexing Section
st.divider()
st.header("📥 Database Indexing")
st.write("Upload new audio files to store their 1024-dimensional embeddings in the SQLite database.")

new_file = st.file_uploader("Choose a .wav file for indexing", type=['wav'], key="index_upload")
if new_file:
    if st.button("Add to Database"):
        with st.spinner('Processing...'):
            files = {"file": (new_file.name, new_file.getvalue(), "audio/wav")}
            res = requests.post("http://api:8000/embeddings", files=files)
            if res.status_code == 200:
                st.success(f"Successfully indexed: **{new_file.name}**")
            else:
                st.error("Failed to index file. Check API logs.")