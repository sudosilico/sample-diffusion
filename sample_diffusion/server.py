from flask import Flask
from flask_socketio import SocketIO

class SocketIOServer:
    def __init__(self, args, host='localhost', port=5001, website_url=None):
        self.args = args
        self.host = host
        self.port = port
        self.website_url = website_url

        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ba254da4-9920-4ce9-aa70-c2cc4cb62658'

        url = f'ws://{host}:{str(port)}'

        socketio = SocketIO(self.app, cors_allowed_origins=[website_url, url], async_mode="threading")

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

        @socketio.on("create_generation")
        def create_generate(data):
            class Object:
                pass

            generation_args = Object()

            generation_args.sample_length_multiplier = data['sample_length_multiplier']
            generation_args.input_sr = data['input_sr']
            generation_args.noise_level = data['noise_level']
            generation_args.n_steps = data['n_steps']
            generation_args.n_samples = data['n_samples']
            generation_args.seed = data['seed']

            if self.ongenerate is not None:
                self.ongenerate(generation_args, socketio)

            return {}

        @socketio.on("create_variation")
        def create_variation(data):
            return {}

        self.socketio = socketio

    def progress_callback(self):
        def callback(cbArgs):
            x = cbArgs.x
            i = cbArgs.i
            t = cbArgs.t
            pred = cbArgs.pred

            self.socketio.emit('generator_progress', { 'x': x, 'i': i, 't': t, 'pred': pred, })
            return
        return callback

    def start(self):
        print("Starting SocketIO server...")
        self.socketio.run(self.app, host=self.host, port=self.port)