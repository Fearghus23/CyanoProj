# Create database structure
import sqlite3

conn = sqlite3.connect('database/counts.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS counts (
                    image_id TEXT PRIMARY KEY,
                    bacteria_counts TEXT)''')
conn.commit()
conn.close()
