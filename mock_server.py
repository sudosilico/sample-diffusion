import argparse
from flask import Flask
from flask_socketio import SocketIO

def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ws_host", type = str, default = 'localhost', help = "host for the websocket server")
    parser.add_argument("--ws_port", type = int, default = 5001, help = "port for the websocket server")
    parser.add_argument("--public_url", type = str, default = 'http://localhost:3000', help = "public url for the dashboard")

    return parser.parse_args()

def main():
    args = parse_cli_args()

    socketio, app = create_socket_server(args)
    
    start_socket_server(socketio, app, args)

def create_socket_server(args):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'ba254da4-9920-4ce9-aa70-c2cc4cb62658'

    ws_url = 'ws://' + args.ws_host + ':' + str(args.ws_port)
    socketio = SocketIO(app, cors_allowed_origins=[args.public_url, ws_url], async_mode="threading")

    @socketio.on("connect")
    def connect(auth):
        global clients
        
        clients += 1
        print("Client connected")
        print("There are " + str(len(clients)) + " clients connected")

    @socketio.on("disconnect")
    def disconnect():
        global clients
        
        clients -= 1
        print("Client disconnected")
        print("There are " + str(len(clients)) + " clients connected")

    @socketio.on("generate")
    def generate():
        return {}

    @socketio.on("regenerate")
    def regenerate():
        return {}

    return socketio, app

def start_socket_server(socketio, app, args):
    print("Starting socketio server...")
    socketio.run(app, host=args.ws_host, port=args.ws_port)

main()