import sqlite3

def save_result(upload_id, prediction, confidence):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO Detection_Result (Upload_ID, Prediction, Confidence)
    VALUES (?, ?, ?)
    """, (upload_id, prediction, confidence))

    conn.commit()
    conn.close()

    return upload_id