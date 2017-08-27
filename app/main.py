from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
import logging
import os
import yaml

import cloudml_client

import requests
import requests_toolbelt.adapters.appengine
requests_toolbelt.adapters.appengine.monkeypatch()

import congressapi
import api_keys
congressapi.set_api_key(api_keys.PROPUBLICA_CONGRESS_API_KEY)

import vote_util

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
app.debug = True

@app.route("/")
def hello():
    return render_template('language-model.html')

@app.route("/generate")
def generate():
    result = {}
    print "Params data:", request.args
    temp = float(request.args.get("temp", 1.0))
    sample = get_cloud_sample(temp)
    result['text'] = sample
    return jsonify(result)

# TODO(kjchavez): Support multiple pages of recent bills.
@app.route("/active_bills")
def active_bills():
    bills = congressapi.recent_bills(congressapi.current_congress(),
                                     congressapi.Chamber.BOTH,
                                     "active")
    return render_template('active-bills.html', bills=bills['bills'])

@app.route("/bill/<bill_id>")
def bill_details(bill_id):
    bill_id, congress_num = bill_id.split('-')
    bill = congressapi.congress.bill(int(congress_num), bill_id)
    votes = list(vote_util.get_input_data(bill))
    predictions = cloudml_client.predict_json('us-legislation-data', 'vote_prediction', votes)
    _add_predictions(votes, predictions)
    return render_template("bill-details.html", bill=bill, examples=votes)

def _add_predictions(votes, preds):
    for i, pred in enumerate(preds):
        votes[i]['predict_aye'] = pred['aye'][0]

@app.route("/predictions/<bill_id>")
def predictions(bill_id):
    bill_id, congress_num = bill_id.split('-')
    bill = congressapi.congress.bill(int(congress_num), bill_id)
    votes = list(vote_util.get_input_data(bill))
    predictions = cloudml_client.predict_json('us-legislation-data', 'vote_prediction', votes)
    _add_predictions(votes, predictions)
    return jsonify({'bill': bill, 'votes': votes})
