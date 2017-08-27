import vote_util
import pprint
import features

bill = {u'active': True,
  u'bill_id': u'hr3353-115',
  u'bill_type': u'hr',
  u'bill_uri': u'https://api.propublica.org/congress/v1/115/bills/hr3353.json',
  u'committee_codes': [u'HSAP'],
  u'committees': u'House Appropriations Committee',
  u'congressdotgov_url': u'https://www.congress.gov/bill/115th-congress/house-bill/3353',
  u'cosponsors': 0,
  u'enacted': None,
  u'govtrack_url': u'https://www.govtrack.us/congress/bills/115/hr3353',
  u'gpo_pdf_uri': None,
  u'house_passage': None,
  u'introduced_date': u'2017-07-21',
  u'latest_major_action': u'Placed on the Union Calendar, Calendar No. 169.',
  u'latest_major_action_date': u'2017-07-21',
  u'number': u'H.R.3353',
  u'primary_subject': u'Economics and Public Finance',
  u'senate_passage': None,
  u'sponsor_id': u'D000600',
  u'sponsor_name': u'Mario Diaz-Balart',
  u'sponsor_party': u'R',
  u'sponsor_state': u'FL',
  u'sponsor_uri': u'https://api.propublica.org/congress/v1/members/D000600.json',
  u'subcommittee_codes': [],
  u'summary': u'',
  u'summary_short': u'',
  u'title': u'Making appropriations for the Departments of Transportation, and Housing and Urban' +
             'Development, and related agencies for the fiscal year ending September 30, 2018, and for other purposes.',
  u'vetoed': None}

for x in vote_util.get_input_data(bill):
    print "======================"
    pprint.pprint(x)

