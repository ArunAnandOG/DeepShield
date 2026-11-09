import sqlite3

def save_upload(file_name, file_path):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO Upload (File_Name, File_Path)
    VALUES (?, ?)
    """, (file_name, file_path))

    conn.commit()

    upload_id = cursor.lastrowid
    conn.close()
    
    return upload_id