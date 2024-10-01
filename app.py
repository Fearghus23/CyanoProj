# app.py
from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    conn = sqlite3.connect('database/counts.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM counts')
    data = cursor.fetchall()
    conn.close()
    counts_list = []
    for row in data:
        image_id = row[0]
        counts = dict(item.split(':') for item in row[1].split(','))
        counts_list.append({'image_id': image_id, 'counts': counts})
    return render_template('index.html', counts_list=counts_list)

if __name__ == '__main__':
    app.run(debug=True)
