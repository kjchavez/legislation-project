"""
    Creates the train/valid/test splits for the vote prediction task.
"""
from __future__ import print_function
import argparse
from datetime import datetime
import logging
import pymongo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_name", '-d', default="congress")
    parser.add_argument("--mongodb_uri", default='mongodb://localhost:27017/')
    parser.add_argument("--output", default='data/votes.dat')
    return parser.parse_args()

_VOTE_DECISION_TABLE = {
    'Yea': 'Aye',
    'Aye': 'Aye',
    'No': 'Nay',
    'Nay': 'Nay',
    'Not Voting': 'Not Voting',
    'Present': 'Present'
}
def normalize(decision):
    return _VOTE_DECISION_TABLE.get(decision, 'Unknown')

def get_examples_for_decision(vote_data, vote_decision):
    members = vote_data['votes'].get(vote_decision, [])
    vote_id = vote_data['vote_id']
    vote_decision = normalize(vote_decision)
    if isinstance(members, str):
        print(members)

    if isinstance(vote_data, str):
        print(vote_data)

    for m in members:
        if not isinstance(m, dict):
            logging.warning("Voting member is not identified properly: %s", m)

    bill = vote_data['bill']
    bill_id = '%s%d-%d' % (bill['type'], bill['number'], bill['congress'])
    return [(m['id'], bill_id, vote_data['vote_id'], vote_decision)
            for m in members if isinstance(m, dict)]

def vote_per_member(vote_data):
    for decision in vote_data['votes'].keys():
        for x in get_examples_for_decision(vote_data, decision):
            yield x

def get_all_votes(cursor):
    for v in cursor:
        for x in vote_per_member(v):
            yield x

def _approximate_age(birthdate_str):
    delta = datetime.now() - datetime.strptime(birthdate_str, '%Y-%m-%d')
    return int(delta.days / 365.25)

def _most_recent_party(member_data):
    return member_data['terms'][-1]['party']

def get_legislator_features(db, bioguide_id):
    if len(bioguide_id) == 4 and bioguide_id.startswith('S'):
        # This is actual a LIS id...
        member_data = db.members.find_one({'id.lis': bioguide_id})
        bioguide_id = member_data['_id']
    else:
        member_data = db.members.find_one({'_id': bioguide_id})

    f = {'id': bioguide_id}
    if not member_data:
        logging.warning("ID %s not found in database", bioguide_id)
        return f
    most_recent_term = member_data['terms'][-1]
    f['state'] = most_recent_term['state']
    f['most_recent_party'] = most_recent_term['party']
    f['age'] = _approximate_age(member_data['bio']['birthday'])
    if most_recent_term['type'] not in ('sen', 'rep'):
        raise Exception("Unknown term type: %s" % most_recent_term['type'])

    f['most_recent_chamber'] = most_recent_term['type']
    return f

def _bill_sponsor_party(db, bill_data):
    if 'sponsor' not in bill_data or bill_data['sponsor'] is None:
        logging.warning("bill_data does not contain a sponsor")
        return None
    if 'thomas_id' not in bill_data['sponsor']:
        logging.warning("Sponsor %s does not have thomas id.", str(bill_data['sponsor']))
        return None

    thomas_id = bill_data['sponsor']['thomas_id']
    sponsor = db.members.find_one({'id.thomas': thomas_id})
    if not sponsor:
        logging.warning("Coulding find sponsor with thomas id: %s", thomas_id)
        return None

    return _most_recent_party(sponsor)


def get_bill_features(db, bill_id):
    f = {'id': bill_id}
    bill_data = db.bills.find_one({'_id': bill_id})
    if not bill_data:
        logging.warning("ID %s not found in database", bill_id)
        return f

    # Most recent party affiliation of the original bill sponsor.
    # NOTE: This is not necessarily their affiliation at time of the bill was voted on.
    sponsor_party = _bill_sponsor_party(db, bill_data)
    if sponsor_party is not None:
        f['sponsor_party'] = sponsor_party

    return f

# TODO(kjchavez): This process is inefficient. We should batch the lookups into the database.
class ExampleGenerator(object):
    def __init__(self, database):
        self.db = database

    def create_example(self, member_id, bill_id, vote_id, decision):
        member = get_legislator_features(self.db, member_id)
        bill = get_bill_features(self.db, bill_id)
        return {'member': member, 'bill': bill, 'aye': decision == 'Aye'}

def main():
    args = parse_args()
    client = pymongo.MongoClient(args.mongodb_uri)
    db = client[args.database_name]
    votes = db.votes.find({'bill.type': {'$in': ['hr', 'hjres', 's', 'sjres']},
                           'category': 'passage'})
    votes.batch_size(100)

    generator = ExampleGenerator(db)
    with open(args.output, 'w') as fp:
        for i, example in enumerate(get_all_votes(votes)):
            if (i + 1) % 1000 == 0:
                print("%d examples written." % i)
            print(generator.create_example(*example), file=fp)


if __name__ == '__main__':
    main()
