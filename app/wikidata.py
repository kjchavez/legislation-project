from SPARQLWrapper import SPARQLWrapper, JSON

PROP_FREEBASE_ID = 'P646' # Freebase Id property
TEST_FREEBASE_ID = "/m/03h4jcq"
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

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def lookup(freebase_id):
    sparql.setQuery(QUERY_TEMPLATE % freebase_id)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results['results']['bindings']
