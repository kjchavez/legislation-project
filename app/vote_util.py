"""

I need a library that can take the results of the /recent_bills endpoint, e.g.

    {
      "active": false, 
      "bill_id": "hr3165-115", 
      "bill_type": "hr", 
      "bill_uri": "https://api.propublica.org/congress/v1/115/bills/hr3165.json", 
      "committee_codes": [
        "HSVR"
      ], 
      "committees": "House Veterans&#39; Affairs Committee", 
      "congressdotgov_url": "https://www.congress.gov/bill/115th-congress/house-bill/3165", 
      "cosponsors": 1, 
      "enacted": "null", 
      "govtrack_url": "https://www.govtrack.us/congress/bills/115/hr3165", 
      "gpo_pdf_uri": "", 
      "house_passage": "", 
      "introduced_date": "2017-07-06", 
      "latest_major_action": "Referred to the House Committee on Veterans&#39; Affairs.", 
      "latest_major_action_date": "2017-07-06", 
      "number": "H.R.3165", 
      "primary_subject": "", 
      "senate_passage": "", 
      "sponsor_id": "C001092", 
      "sponsor_uri": "https://api.propublica.org/congress/v1/members/C001092.json", 
      "subcommittee_codes": [], 
      "summary": "", 
      "summary_short": "", 
      "title": "To authorize the Secretary of Veterans Affairs to carry out a pilot program to provide grants to veterans service organizations for upgrading local chapter facilities, including technology at such facilities, in rural areas, and for other purposes.", 
      "vetoed": "null"
    }

and produce the proper input features for the vote prediction model. This will likely have to query
the Congress API live, since the data may not have been integrated into our local database yet.
"""

import congressapi
import api_keys
import features
import data_util
from google.appengine.api import urlfetch
import logging
import json
import pprint
import functools

# When looking at the landing page for a bill, we will load (and make predictions)
# on at most this many congress members.
MAX_MEMBERS_TO_LOAD = 10

def _get_congress_api_urls(urls):
    """ Retrieves data in parallel from ProPublica Congress API. """
    results = []
    def handle_result(rpc):
        result = rpc.get_result()
        if result.status_code == 200:
            results.append(json.loads(result.content))
            logging.info('Added json data')
        else:
            logging.info('Error in fetch.')

    rpcs = []
    logging.info("Fetching data for %d urls", len(urls))
    for i, uri in enumerate(urls):
        rpc = urlfetch.create_rpc()
        rpc.callback = functools.partial(handle_result, rpc)
        urlfetch.make_fetch_call(rpc, uri, headers={'X-API-Key':
                                                    api_keys.PROPUBLICA_CONGRESS_API_KEY})
        rpcs.append(rpc)


    for rpc in rpcs:
        rpc.wait()

    logging.info("Data fetch complete.")
    return results


def _get_chamber(bill):
    if bill['bill_type'].startswith('h'):
        return 'house'
    elif bill['bill_type'].startswith('s'):
        return 'senate'
    else:
        return 'unknown'

def _get_congress(bill):
    return int(bill['bill_id'].split('-')[-1])

def get_voting_members(bill):
    chamber = _get_chamber(bill)
    members = congressapi.members(_get_congress(bill), chamber)[0]['members']

    # Note that member['api_uri'] has URLs such as
    # 'https://api.propublica.org/congress/v1/members/A000055.json'
    return members

def _normalize_party(party):
    if party == 'R':
        return 'Republican'
    elif party == 'D':
        return 'Democrat'
    else:
        return 'Unknown'

def get_sponsor_data(bill):
    if 'sponsor_id' in bill:
        return congressapi.member(bill['sponsor_id'])
    else:
        return None

def expand(bill_data):
    """ Expands bill data to have all necessary information for making predictions about a roll call
    vote.
    """
    data = {'bill' : bill_data}

    # Get information of all members who will vote on this bill
    data['voters'] = get_voting_members(bill_data)

    # Get information about the bill's primary sponsor
    data['sponsor'] = get_sponsor_data(bill_data)
    return data

def _split(data):
    for voter in data['voters']:
        yield {'bill' : data['bill'], 'voter': voter, 'sponsor' : data['sponsor']}

def get_input_data(bill):
    """ Returns inputs for the vote_prediction model given the bill JSON returned by ProPublica
    /recent_bills endpoint. """
    for x in _split(expand(bill)):
        # logging.info(pprint.pformat(x))
        yield data_util.extract_infer_features(x, features.FEATURES)

