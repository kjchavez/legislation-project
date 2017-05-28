from flask import Flask
from flask import jsonify
from flask import render_template

from grpc.beta import implementations
import numpy as np

import tensorflow as tf
import tfserving_client as client

hostport = "127.0.0.1:9000"
host, port = hostport.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = client.beta_create_PredictionService_stub(channel)

def get_probs(token):
    request = client.PredictRequest()
    request.model_spec.name = 'lm'
    request.model_spec.signature_name = "next_token"
    request.inputs['input_token'].CopyFrom(
        tf.contrib.util.make_tensor_proto([[token]], shape=[1, 1]))

    response = stub.Predict(request, 5.0)
    probs = list(response.outputs['probs'].float_val)
    return probs

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/query")
def query():
    result = {'probs': get_probs(5)}
    return jsonify(result)
