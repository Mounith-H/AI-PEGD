import serial
import json
import threading
from flask import Flask, jsonify, send_from_directory
import serial.tools.list_ports

app = Flask(__name__)
gunshot_data = []  # Store gunshot data

# ===== SERIAL PORT SELECTION =====
ports = list(serial.tools.list_ports.comports())
for i, p in enumerate(ports):
    print(f"{i}: {p.device} ({p.description})")
port_index = int(input("Select the port index: "))
selected_port = ports[port_index].device

# ===== SERIAL READING THREAD =====
def read_serial():
    global gunshot_data
    try:
        ser = serial.Serial(selected_port, 9600, timeout=1)
        print(f"📡 Reading from {selected_port}...")
        while True:
            line = ser.readline().decode().strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                print("Received:", data)
                gunshot_data.append(data)

                # Optional: Limit list to last 500 entries
                if len(gunshot_data) > 500:
                    gunshot_data = gunshot_data[-500:]
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON:", line)
    except Exception as e:
        print("❌ Serial read error:", e)

# Start reading in background
threading.Thread(target=read_serial, daemon=True).start()

# ===== ROUTES =====
@app.route('/api/gunshots')
def get_gunshots():
    return jsonify(gunshot_data[-50:])  # Send only recent 50 gunshots

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'dashboard.html')

# ===== START FLASK SERVER =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
