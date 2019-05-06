import csv
import json

j = json.load(open('countries.json', 'r'))

def clean(x):
    x = x.replace(' ', '_')
    x = x.replace('.','_')
    x = x.replace('"', '')
    x = x.replace("'", '')
    x = x[0].lower() + x[1:]
    return x

triples = []
cca3toname = {}
for c in j:
    cca3toname[c['cca3']] = clean(c['name']['common'])

for c in j:
    try:
        name = clean(c['name']['common'])
        capital = clean(c['capital'][0])
        subregion = clean(c['subregion'])
        region = clean(c['region'])
    except:
        continue

    for neighbor in c['borders']:
        triples.append('{},borders,{}'.format(name, cca3toname[neighbor]))
    triples.append('{},capital_of,{}'.format(capital, name))
    triples.append('{},in_subregion,{}'.format(name, subregion))
    triples.append('{},in_region,{}'.format(name, region))
    if not '{},is_subregion_of,{}'.format(subregion, region) in triples:
        triples.append('{},is_subregion_of,{}'.format(subregion, region))

w = csv.writer(open('tripout.csv', 'w'))
w.writerows([(x.split(',')[0],x.split(',')[1],x.split(',')[2]) for x in triples])
w.close()