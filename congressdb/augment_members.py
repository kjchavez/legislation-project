"""
Augments 'members' collection with information from wikidata.

Use Python 3 for asyncio

"""

import argparse
import glob
import json
import logging
import os
from os.path import join
import sys
import pymongo
import yaml
import pprint
from SPARQLWrapper import SPARQLWrapper, JSON
from multiprocessing.dummy import Pool


sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
QUERY_TEMPLATE = \
"""
SELECT ?human ?humanLabel ?school ?schoolLabel
WHERE
{
        ?human wdt:P31 wd:Q5 .
        ?human wdt:P646 "%s" .
        OPTIONAL { ?human wdt:P69 ?school }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }
}
"""
def lookup(freebase_id):
    sparql.setQuery(QUERY_TEMPLATE % freebase_id)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return results['results']['bindings'][0]
    except:
        logging.warning("Failed on freebase_id: %s", freebase_id)
        return None

path = os.path.dirname(os.path.realpath(__file__))
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_name", '-d', default="congress")
    parser.add_argument("--mongodb_uri", default='mongodb://localhost:27017/')
    parser.add_argument("--num_pools", '-p', type=int, default=10)
    parser.add_argument("--batch_size", '-b', type=int, default=100)
    parser.add_argument('--force_refresh', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()

def load_single(m):
    if 'google_entity_id' not in m['id']:
        return None

    freebase_id = m['id']['google_entity_id'][3:]
    return (m['_id'], lookup(freebase_id))

def load_wikidata(members, num_pools=10):
    p = Pool(num_pools)
    entries = p.map(load_single, members)
    return dict(e for e in entries if e)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    client = pymongo.MongoClient(args.mongodb_uri)
    db = client[args.database_name]
    print "-"*80
    print "Sample entry in the members database:"
    print "-"*80
    pprint.pprint(db.members.find_one({}))
    print "-"*80
    print
    if args.force_refresh:
        query = {}
    else:
        query = {'wikidata' : {'$exists': False}}

    m = db.members.find(query, {'_id': 1, 'id': 1})
    m = list(m)
    for i in xrange(0, len(m), args.batch_size):
        wikidata = load_wikidata(m[i:min(i+args.batch_size, len(m))], num_pools=args.num_pools)
        bulk = db.members.initialize_unordered_bulk_op()
        for idx, data in wikidata.items():
            if not args.dry_run:
                bulk.find({'_id': idx}).update({'$set' : {'wikidata' : data}})

        logging.info("Added updates for %s.", wikidata.keys())
        if not args.dry_run:
            logging.info("Starting commit.")
            bulk.execute()
            logging.info("Committed batch (%d - %d).", i+1, i+args.batch_size)

if __name__ == "__main__":
    main()
