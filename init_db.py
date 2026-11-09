import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Upload table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Upload (
    Upload_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    File_Name TEXT NOT NULL,
    File_Path TEXT NOT NULL,
    Upload_Time DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Detection_Result table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Detection_Result (
    Result_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Upload_ID INTEGER,
    Prediction TEXT NOT NULL,
    Confidence REAL NOT NULL,
    Processed_Time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (Upload_ID) REFERENCES Upload(Upload_ID)
)
""")

conn.commit()
conn.close()

print("Tables created successfully!")

def get_filepath_from_db(upload_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT File_Path FROM Upload WHERE Upload_ID = ?
    """, (upload_id,))

    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]   # File_Path
    else:
        return None
    
def get_result_from_db(upload_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT Prediction, Confidence 
        FROM Detection_Result 
        WHERE Upload_ID = ?
    """, (upload_id,))

    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0], result[1]   # prediction, confidence
    else:
        return None, None