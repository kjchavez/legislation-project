from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

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

def get_sample(temp):
    request = client.PredictRequest()
    request.model_spec.name = 'lm'
    request.model_spec.signature_name = "GenerateSample"
    request.inputs['temp'].CopyFrom(
        tf.contrib.util.make_tensor_proto([temp], shape=[], dtype=tf.float32))
    response = stub.Predict(request, 5.0)
    token_ids = list(response.outputs['tokens'].int_val)
    return token_ids

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def hello():
    return render_template('language-model.html')

@app.route("/query")
def query():
    result = {'probs': get_probs(5)}
    return jsonify(result)

count = 1
@app.route("/generate")
def generate():
    result = {}
    print "Params data:", request.args
    temp = float(request.args.get("temp", 1.0))
    sample = get_sample(temp)
    # TODO(kjchavez): Need vocabulary!
    result['text'] = str(sample) 
    return jsonify(result)
