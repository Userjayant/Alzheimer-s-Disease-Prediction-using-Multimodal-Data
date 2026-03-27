import sqlite3

def init_db():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age REAL,
        mmse REAL,
        cdr REAL,
        prediction TEXT,
        confidence REAL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def insert_data(age, mmse, cdr, prediction, confidence):
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO history (age, mmse, cdr, prediction, confidence)
    VALUES (?, ?, ?, ?, ?)
    """, (age, mmse, cdr, prediction, confidence))

    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect("patients.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY date DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows