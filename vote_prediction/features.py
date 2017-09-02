from datetime import datetime
from data_util import Feature

def _most_recent_party(member_data):
    return member_data['terms'][-1]['party']

def _most_recent_term(member_data):
    return member_data['terms'][-1]

class VoterState(Feature):
    default = "UNK"
    def f_train(self, x):
        return str(_most_recent_term(x['member'])['state'])

    def f_infer(self, x):
        return str(x['voter']['state'])

class VoterAge(Feature):
    default = 0
    def f_train(self, x):
        birthdate_str = x['member']['bio']['birthday']
        delta = datetime.now() - datetime.strptime(birthdate_str, '%Y-%m-%d')
        return int(delta.days / 365.25)

    def f_infer(self, x):
        birthdate_str = x['voter']['date_of_birth']
        delta = datetime.now() - datetime.strptime(birthdate_str, '%Y-%m-%d')
        return int(delta.days / 365.25)

def normalize_party(text):
    REPUBLICAN = 'republican'
    DEMOCRAT = 'democrat'
    lower = text.lower()
    if lower.startswith('r'):
        return REPUBLICAN
    if lower.startswith('d'):
        return DEMOCRAT
    return lower

class VoterParty(Feature):
    default = 'UNK'
    def f_train(self, x):
        return normalize_party(_most_recent_term(x['member'])['party'])

    def f_infer(self, x):
        return normalize_party(x['voter']['party'])

class VoterChamber(Feature):
    default = 'UNK'
    def f_train(self, x):
        return _most_recent_term(x['member'])['type']

    def f_infer(self, x):
        house = 'district' in x['voter'].keys()
        if house:
            return 'rep'
        else:
            return 'sen'

class SponsorParty(Feature):
    default = 'UNK'
    def f_train(self, x):
        return normalize_party(_most_recent_party(x['sponsor']))

    def f_infer(self, x):
        return normalize_party(x['bill']['sponsor_party'])

class BillTitle(Feature):
    default = 'UNK'
    def f_train(self, x):
        return x['bill']['official_title']

    def f_infer(self, x):
        return x['bill']['title']

class BillId(Feature):
    def f_train(self, x):
        return x['bill']['_id']

    def f_infer(self, x):
        return x['bill']['bill_id']

class Decision(Feature):
    default = 'Nay'
    def f_train(self, x):
        return x['decision']

FEATURES = [
    #BillId(),
    BillTitle(),
    SponsorParty(),
    #VoterChamber(),
    VoterParty(),
    VoterAge(),
    VoterState(),
]

LABELS = [
    Decision(),
]
