import requests
import itertools

BASE_URL = "http://congress.api.sunlightfoundation.com/"

bill_filters = [
    'bill_id',
    'bill_type',
    'number',
    'congress',
    'chamber',
    'introduced_on',
    'last_action_at',
    'last_vote_at',
    'last_version_on'
]

def bills(**kwargs):
    assert all(key in bill_filters for key in kwargs.keys()), \
            "Invalid filters. Please use: " + ','.join(bill_filters)

    url = BASE_URL + 'bills'
    if kwargs:
        url += '?' + '&'.join('%s=%s' % (k, v) for k, v in kwargs.items())

    print "URL:", url
    r = requests.get(url)
    return r.json()

def parse(bill_id):
    head, tail = bill_id.split('-', 1)
    congress_number = int(tail)
    bill_type = "".join(itertools.takewhile(str.isalpha, head))
    bill_number = int(head[len(bill_type):])
    return (bill_type, bill_number, congress_number)


BILL_TEXT_URL = "https://www.congress.gov/bill/%dth-congress/%s/%d/text?format=txt"
DIR_FOR_TYPE = {
    'sres': 'senate-resolution',
    's': 'senate-bill'
}
def bill_url(bill_id):
    bill_type, bill_number, congress_number = parse(bill_id)
    assert bill_type in DIR_FOR_TYPE, 'Unknown bill type: %s' % bill_type
    return BILL_TEXT_URL % (congress_number, DIR_FOR_TYPE[bill_type], \
                            bill_number)

START_TAG = '<pre id="billTextContainer">'
END_TAG = '</pre>'
def extract_bill_text(html_content):
    idx = html_content.find(START_TAG)
    if idx == -1:
        return "NOT FOUND"

    end_idx = html_content.find(END_TAG, idx)
    if end_idx == -1:
        raise Exception("Malformed HTML content")

    return html_content[idx+len(START_TAG):end_idx]

def get_bill_text(bill_id):
    url = bill_url(bill_id)
    r = requests.get(url)
    return extract_bill_text(r.content)
