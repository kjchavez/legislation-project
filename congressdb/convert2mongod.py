"""
    Converts the downloaded Congress bulk data into a MongoDB database.
"""
import argparse
import glob
import json
import logging
import os
import sys
import pymongo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--congress_data", required=True)
    parser.add_argument("--database_name", '-d', default="congress")
    parser.add_argument("--mongodb_uri", default='mongodb://localhost:27017/')
    parser.add_argument("--bills", action="store_true")
    parser.add_argument("--votes", action="store_true")
    parser.add_argument("--reset", action="store_true", help="if true, drops "
                        "the mentioned collections before starting to "
                        "populate.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()

def _get_text_versions(path):
    versions = []
    for version_dir in glob.glob(os.path.join(path, 'text-versions', '*')):
        with open(os.path.join(version_dir, "data.json")) as fp:
            data = json.load(fp)

        with open(os.path.join(version_dir, 'document.txt')) as fp:
            data['document'] = fp.read()

        versions.append(data)

    return versions

def _batched_insert(path, collection, augment_fn=None,
        batch=1000, json_filename="data.json",
        check_keys=True, dry_run=False):
    """ Inserts json objects that match path/data.json into collection.

    Args:
        path: path to directories that contain json files, may use wildcards.
        collection: a MongoDB collection to insert objects into.
        augment_fn: function (dict, path) -> bool that augments the dict before
                    inserting it into collection. If this function returns
                    false, the object is skipped.
        batch: number of objects to add to collection at a time. They will be
               stored in memory until added.
        json_filename: name of the file with json data, usually data.json
        dry_run: if true, nothing actually gets added to database.
    """
    data = []
    for i, d in enumerate(glob.glob(path)):
        if (i+1) % batch == 0:
            if not dry_run and data:
                try:
                    collection.insert_many(data)
                except Exception, e:
                    failed_batch_filename = "/tmp/convert2mongod.failure.%d-%d.txt" \
                                             % (i - batch, i)
                    logging.error("Failed to insert a batch. Some elements " +
                                  "may have not been written. See %s.",
                                  failed_batch_filename)
                    with open(failed_batch_filename, 'w') as fp:
                        print >> fp, '\n'.join(str(x) for x in data)
            else:
                logging.info(str(elem))

            del data
            data = []
            logging.info("Processed %d..." % (i+1))

        elem = {}
        if not os.path.isfile(os.path.join(d, json_filename)):
            logging.warning("Missing data.json file for elem %s",
                    os.path.basename(d))
            continue

        with open(os.path.join(d, json_filename)) as fp:
            elem = json.load(fp)

        if augment_fn is not None:
            if not augment_fn(elem, d):
                continue

        data.append(elem)

    if data:
        collection.insert_many(data)
        del data

def _augment_bill(bill, d):
    if 'bill_id' not in bill:
        logging.warning("Warning: bill does not have id, skipping!")
        return False

    bill['_id'] = bill['bill_id']
    bill['text_versions'] = _get_text_versions(d)
    return True

# TODO(kjchavez): If this is too slow, parallelize.
def create_bills_collection(congress_data, mdb, batch=1000, dry_run=False):
    path = os.path.join(congress_data, '*', 'bills', '*', '*')
    _batched_insert(path, mdb.bills, augment_fn=_augment_bill, batch=batch,
                    dry_run=dry_run)
    logging.info("Creating index over bill text...")
    mdb.bills.create_index([('text_versions.document', pymongo.TEXT)])

def _escape(key):
    escaped = key.replace('.', '[dot]')
    return escaped

def _augment_vote(vote, d):
    if 'vote_id' not in vote:
        logging.warning("Vote does not have id, skipping!")
        return False

    if 'votes' in vote:
        for key in vote['votes']:
            escaped_key = _escape(key)
            if escaped_key != key:
                logging.info("Escaping key: %s", key)
                vote['votes'][escaped_key] = vote['votes'].pop(key)

    # vote['_id'] = vote['vote_id']
    return True

def create_votes_collection(congress_data, mdb, batch=1000, dry_run=False):
    path = os.path.join(congress_data, '*', 'votes', '*', '*')
    _batched_insert(path, mdb.votes, augment_fn=_augment_vote, batch=batch,
                    dry_run=dry_run, check_keys=False)
    logging.info("Creating index over vote text...")
    mdb.votes.create_index([('question', pymongo.TEXT), ('subject', pymongo.TEXT)])


def main():
    args = parse_args()
    client = pymongo.MongoClient(args.mongodb_uri)
    db = client[args.database_name]

    if args.bills:
        if args.reset:
            db.bills.drop()
        logging.info("BEGIN: Creating 'bills' collection.")
        create_bills_collection(args.congress_data, db, dry_run=args.dry_run)
        logging.info("END: Creating 'bills' collection.")

    if args.votes:
        if args.reset:
            db.votes.drop()
        logging.info("BEGIN: Creating 'votes' collection.")
        create_votes_collection(args.congress_data, db, dry_run=args.dry_run)
        logging.info("END: Creating 'votes' collection.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
