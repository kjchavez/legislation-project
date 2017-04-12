"""

For more information about congressional bills, see:

	https://www.gpo.gov/help/about_congressional_bills.htm

"""
import codecs
from enum import Enum
import glob
import os

class BillType(Enum):
    HOUSE_CONCURRENT_RES = 'hconres'
    HOUSE_JOINT_RES = 'hjres'
    HOUSE_RES = 'hres'
    HOUSE_BILL = 'hr'

    SENATE_CONCURRENT_RES = 'sconres'
    SENATE_JOINT_RES = 'sjres'
    SENATE_RES = 'sres'
    SENATE_BILL = 's'


# TODO(kjchavez): There are many other versions missing from this list.
class BillVersion(Enum):
    # Senate
    INTRODUCED_SENATE = 'is'
    ENGROSSED_SENATE = 'es'
    PLACED_ON_CALENDAR_SENATE = 'pcs'
    FAILED_PASSAGE_SENATE = 'fps'

    # House
    INTRODUCED_HOUSE = 'ih'
    ENGROSSED_HOUSE = 'eh'
    PLACED_ON_CALENDAR_HOUSE = 'pch'
    FAILED_PASSAGE_HOUSE = 'fph'

    # Joint
    ENROLLED_BILL = 'enr'

def _bill_id_from_path(path):
    parts = path.split('/')
    assert len(parts) >= 7
    assert parts[-1] == 'document.txt'
    version = parts[-2]
    assert parts[-3] == 'text-versions'
    bill_id = parts[-4]
    bill_type = parts[-5]  # included in id
    assert parts[-6] == 'bills'
    congress = parts[-7]
    return "%s-%s.%s" % (bill_id, congress, version)


class CongressDatabase(object):
    """ Interface for querying Congress legislative data.

    Currently, simply provides a wrapper over file access, but it might make
    sense to put the data in a real data base for faster query times.
    """
    def __init__(self, path):
        self.base_path = path

    def bill_text(self, congress_num='*', bill_type='*', version='*'):
        """ Returns iterator over full text of bills from a Congress. """
        if isinstance(bill_type, BillType):
            bill_type = bill_type.value
        if isinstance(version, BillVersion):
            version = version.value

        pattern = os.path.join(self.base_path,
                "%s/bills/%s/*/text-versions/%s/document.txt" %
                (congress_num, bill_type, version))

        for filename in glob.glob(pattern):
            _id = _bill_id_from_path(filename)
            with codecs.open(filename, encoding='utf-8') as fp:
                yield (_id, fp.read())
