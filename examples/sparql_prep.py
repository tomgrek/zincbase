"""This file recursively builds a CSV file of triples from Wikidata, suitable for import
into Zincbase. It's quite rough and ready, and currently starts from one hardcoded query ('all entites
that are rock bands'), expanding out once to learn everything it can about each band (e.g., its members)
and then again to learn everything it can about that (e.g. its members' birthplaces).

Note: requires `pip install sparqlwrapper`"""

import argparse
import csv
import json
import tqdm

from SPARQLWrapper import SPARQLWrapper, JSON



def query_ent(ent, level=0, max_level=1):
    if level >= max_level:
        return None
    sparql.setQuery("""
    PREFIX entity: <http://www.wikidata.org/entity/>
    SELECT ?propUrl ?propLabel ?valUrl ?valLabel
    WHERE
    {
        hint:Query hint:optimizer 'None' .
	{	BIND(entity:%s AS ?valUrl) .
		BIND("N/A" AS ?propUrl ) .
		BIND("identity"@en AS ?propLabel ) .
	}
	UNION
	{	entity:%s ?propUrl ?valUrl .
		?property ?ref ?propUrl .
		?property rdf:type wikibase:Property .
		?property rdfs:label ?propLabel
	}

  	?valUrl rdfs:label ?valLabel
	FILTER (LANG(?valLabel) = 'en') .
	FILTER (lang(?propLabel) = 'en' )
    }
    """ % (ent, ent))
    results = sparql.query().convert()
    results = results['results']['bindings']
    for result in results:
        try:
            prop = result['propUrl']['value'].split('/')[-1]
            if prop=='A':
                continue
            proplabel = result['propLabel']['value']
            dic.write(prop + '\t' + proplabel + '\n')
            ob = result['valUrl']['value'].split('/')[-1]
            oblabel = result['valLabel']['value']
            dic.write(ob + '\t' + oblabel + '\n')
            c.writerow((ent, prop, ob))
            if level < max_level:
                if not ob in already:
                    query_ent(ob, level=level+1, max_level=max_level)
                    already[ob] = True
        except:
            pass

if __name__ == '__main__':
    print("This file builds a graph by querying Wikidata from some starting point.")
    print("""
Example: `python sparql_prep.py --tsv_filename_out outfile.tsv --dict_filename_out dictfile.txt \
--max_levels 2 --starting_predicate P31 --starting_object Q5741069`

This would build a graph beginning with any entity that's an instance of a rock band - for each rock band,
get all of their linked properties and add to graph, and for each linked property get all of its linked
properties and add to graph (ie 2 levels deep.)
    """)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_filename_out", type=str, default="triples_out.tsv", help="Filename to output triples to, as TSV.")
    parser.add_argument("--dict_filename_out", type=str, default="dict_out.txt", help="Filename to output dictionary mapping (ie Q numbers to labels)")
    parser.add_argument("--max_levels", type=int, default=2, help="How many levels deep to recurse getting more graph nodes")
    parser.add_argument("--starting_subject", type=str, default=None, help="Q ID of subject to start building graph from")
    parser.add_argument("--starting_predicate", type=str, default='P31', help="P ID of predicate to start building graph from")
    parser.add_argument("--starting_object", type=str, default=None, help="Q ID of object to start building graph from")
    args = parser.parse_args()
    
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    if not args.starting_subject:
        query = "?item wdt:%s wd:%s" % (args.starting_predicate, args.starting_object)
    elif not args.starting_object:
        query = "wd:%s wdt:%s ?item" % (args.starting_subject, args.starting_predicate)
    else:
        print("Please only specify one of either starting subject, or starting object")
        import sys; sys.exit(0)

    sparql.setQuery("""
    SELECT ?item ?itemLabel
    WHERE
    {
        %s .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    """ % query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results = results['results']['bindings']

    cf = open(args.tsv_filename_out, 'w')
    c = csv.writer(cf, delimiter='\t')
    dic = open(args.dict_filename_out, 'w')

    already = {}

    for result in results:
        entity = result['item']['value'].split('/')[-1]
        label = result['itemLabel']['value']
        try:
            dic.write(entity + '\t' + label + '\n')
        except:
            pass
        if not args.starting_subject:
            c.writerow((entity, args.starting_predicate, args.starting_object))
        else:
            c.writerow((args.starting_subject, args.starting_predicate, entity))
        query_ent(entity, max_level=args.max_levels)

    cf.close()
    dic.close()