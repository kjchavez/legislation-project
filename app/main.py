from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

import cloudml_client


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

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def hello():
    return render_template('language-model.html')

@app.route("/generate")
def generate():
    result = {}
    print "Params data:", request.args
    temp = float(request.args.get("temp", 1.0))
    sample = get_cloud_sample(temp)
    # TODO(kjchavez): Need vocabulary!
    result['text'] = sample
    return jsonify(result)
