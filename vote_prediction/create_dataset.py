"""
    Creates the train/valid/test splits for the vote prediction task.

    We have a situation where:
        1. We have a way to iterate over all elements of the dataset, one by one.
        2. The entire dataset doesn't fit on disk.
        3. We're content to just extract some summary f(x) of the data point x.
        4. {f(x) : x in dataset } *does* fit on disk.


    This file helps us create those data sets.

"""
from __future__ import print_function
import argparse
from datetime import datetime
import time
import logging
import pymongo
from data_util import process_data
from features import FEATURES, LABELS
import pprint


# Data generator should make the union of all data sources and make use of the cache to be more
# effective.

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


class BillCache(object):
    cache = {}

    @staticmethod
    def get_bill(db, bill_id):
        if bill_id in BillCache.cache:
            return BillCache.cache[bill_id]
        else:
            bill_data = db.bills.find_one({'_id': bill_id})
            if not bill_data:
                logging.warning("ID %s not found in database", bill_id)
                bill_data = {'_id': bill_id}

            # Yes, intentionally clear the entire cache
            BillCache.cache = {bill_id: bill_data}
            return bill_data


class LegislatorCache(object):
    cache = {}

    @staticmethod
    def get_member_data(db, bioguide_id):
        if len(bioguide_id) == 4 and bioguide_id.startswith('S'):
            # This is actual a LIS id...
            member_data = db.members.find_one({'id.lis': bioguide_id})
            bioguide_id = member_data['_id']
        else:
            member_data = db.members.find_one({'_id': bioguide_id})

        if not member_data:
            logging.warning("ID %s not found in database", bioguide_id)
            return {'_id': bioguide_id}

        return member_data

    @staticmethod
    def get_member(db, member_id):
        if member_id in LegislatorCache.cache:
            return LegislatorCache.cache[member_id]

        data = LegislatorCache.get_member_data(db, member_id)
        LegislatorCache.cache[member_id] = data
        return data

def _bill_sponsor(db, bill_data):
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

    return sponsor

def recursive_keys(x):
    keys = {}
    for key in x:
        if isinstance(x[key], dict):
            keys[key] = recursive_keys(x[key])
        else:
            keys[key] = {}

    return keys


def vote_data_generator(database, batch_size=100):
    """ Generates samples from the vote data by joining member, bill, and vote data from the given
    database.
    """
    votes = database.votes.find({'bill.type': {'$in': ['hr', 'hjres', 's', 'sjres']},
                           'category': 'passage'})
    votes.batch_size(batch_size)
    is_first = True
    for member_id, bill_id, vote_id, decision in get_all_votes(votes):
        member = LegislatorCache.get_member(database, member_id)
        bill = BillCache.get_bill(database, bill_id)
        sponsor = _bill_sponsor(database, bill)
        x = {'member': member, 'bill': bill, 'sponsor': sponsor,'decision': decision}
        if is_first:
            logging.info("Example data point:\n%s", pprint.pformat(recursive_keys(x)))
            is_first = False
        yield x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_name", '-d', default="congress")
    parser.add_argument("--mongodb_uri", default='mongodb://localhost:27017/')
    parser.add_argument("--output", default='data/votes.dat')
    parser.add_argument("--nmax", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    client = pymongo.MongoClient(args.mongodb_uri)
    db = client[args.database_name]

    generator = vote_data_generator(db)
    logging.basicConfig(level=logging.INFO)
    process_data(generator, FEATURES+LABELS,
                 args.output, max_elem=args.nmax)

if __name__ == '__main__':
    main()
