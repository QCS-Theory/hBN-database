from flask import Flask, request, send_file, jsonify, render_template_string, Response
import geoip2.database
import mysql.connector
from datetime import datetime
import io

app = Flask(__name__)

# Configuration - adjust these
GEOIP_DB_PATH = 'GeoLite2-Country.mmdb'
DB_CONFIG = {
    'host': 'database-5017920693.webspace-host.com',
    'user': 'dbu190527',
    'password': 'QuantumCommSys2024!',
    'database': 'dbs14266576',
}

# Initialize GeoIP reader
geo_reader = geoip2.database.Reader(GEOIP_DB_PATH)

# Tracker endpoint: logs visit and returns a 1×1 transparent GIF
@app.route('/track')
def track():
    # Get client IP (map IPv6 loopback to IPv4)
    ip = request.remote_addr or '0.0.0.0'
    if ip == '::1':
        ip = '127.0.0.1'

    try:
        rec = geo_reader.country(ip)
        country_code = rec.country.iso_code or 'Unknown'
        country_name = rec.country.name or 'Unknown'
    except Exception:
        country_code = 'Unknown'
        country_name = 'Unknown'

    # Insert into DB
    cnx = mysql.connector.connect(**DB_CONFIG)
    cursor = cnx.cursor()
    cursor.execute(
        "INSERT INTO visits (ip, country_code, country_name, visited_at) VALUES (%s, %s, %s, %s)",
        (ip, country_code, country_name, datetime.utcnow())
    )
    cnx.commit()
    cursor.close()
    cnx.close()

    # Prepare a fresh 1x1 GIF per request
    pixel_bytes = (
        b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00'
        b'\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00'
        b'\x00\x00\x01\x00\x01\x00\x00\x02\x02L\x01\x00;'
    )
    return Response(pixel_bytes, mimetype='image/gif')

# Stats endpoint - HTML + JSON
@app.route('/stats')
def stats():
    cnx = mysql.connector.connect(**DB_CONFIG)
    cursor = cnx.cursor(dictionary=True)
    cursor.execute('SELECT COUNT(*) AS total FROM visits')
    total = cursor.fetchone()['total']

    cursor.execute(
        'SELECT country_name, country_code, COUNT(*) AS cnt '
        'FROM visits GROUP BY country_code, country_name ORDER BY cnt DESC'
    )
    rows = cursor.fetchall()

    # JSON API support
    if request.args.get('json'):
        return jsonify({'total': total, 'countries': rows})

    # Render HTML
    template = '''
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>Visitor Stats</title></head>
    <body>
      <h1>Total Visits: {{ total }}</h1>
      <h2>By Country</h2>
      <table border="1"><tr><th>Country</th><th>Code</th><th>Visits</th></tr>
      {% for r in rows %}
        <tr><td>{{ r.country_name or 'Unknown' }}</td><td>{{ r.country_code or '--' }}</td><td>{{ r.cnt }}</td></tr>
      {% endfor %}
      </table>
    </body>
    </html>
    '''
    return render_template_string(template, total=total, rows=rows)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
