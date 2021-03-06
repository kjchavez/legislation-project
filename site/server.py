from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from grpc.beta import implementations
import cloudml_client
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


def _format_for_html(tokens):
    # TODO(kjchavez): This is not the nicest format. We should probably do
    # things like: ( a ) -> (a)
    text = ' '.join(tokens)
    r = '<br />'
    text.replace('\r\n',r).replace('\n\r',r).replace('\r',r).replace('\n',r)
    return text

def get_cloud_sample(temp):
    results = cloudml_client.predict_json(
                'us-legislation-data',
                'language_model',
                {'temp': temp})
    return _format_for_html(results[0]['tokens'])

def get_sample(temp):
    request = client.PredictRequest()
    request.model_spec.name = 'lm'
    request.model_spec.signature_name = "GenerateSample"
    request.inputs['temp'].CopyFrom(
        tf.contrib.util.make_tensor_proto([temp], shape=[], dtype=tf.float32))
    response = stub.Predict(request, 5.0)
    tokens = list(response.outputs['tokens'].string_val)
    return _format_for_html(tokens)

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
    sample = get_cloud_sample(temp)
    # TODO(kjchavez): Need vocabulary!
    result['text'] = sample
    return jsonify(result)
