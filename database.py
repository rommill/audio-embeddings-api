import sqlite3
import json

def init_db():
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            embedding TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_embedding(filename, embedding):
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    
    embedding_json = json.dumps(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
    cursor.execute('INSERT INTO audio_data (filename, embedding) VALUES (?, ?)', (filename, embedding_json))
    conn.commit()
    conn.close()

def get_all_embeddings():
    conn = sqlite3.connect('embeddings.db')
    cursor = conn.cursor()
    cursor.execute('SELECT filename, embedding FROM audio_data')
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "filename": row[0],
            "embedding": json.loads(row[1]) 
        })
    return results

init_db()