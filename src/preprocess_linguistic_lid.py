import argparse
import gzip
import json
import csv
import pycountry

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tweet_file", dest="tweet_file", help="Tweet file")
    parser.add_argument("-p", "--properties_file", dest="properties_file", help="Language properties file")
    parser.add_argument("-o", "--output_file", dest="output_file", help="Output file")
    args = parser.parse_args()

    wals_langs = set()
    tweet_langs = set()
    keys = ["family", 
            #"iso_code", 
            #"wals_code", 
            #"countrycodes", 
            "macroarea", 
            #"Name", 
            #"glottocode", 
            "genus", 
            #"141A Writing Systems"
    ]
    languages = {}
    genuses = {}
    families = {}
    macroareas = {}
    
    with gzip.open(args.properties_file, "rt") as ifd:
        for row in csv.DictReader(ifd):
            iso_code = row["iso_code"]
            lang = pycountry.languages.get(alpha_3=iso_code)
            if hasattr(lang, "alpha_2"):
                lang_code = lang.alpha_2
                wals_langs.add(lang_code)
                languages[lang_code] = {"id" : lang_code,
                                        "entity_type" : "language",
                                        "language_to_genus" : row["genus"],
                                        #"language_to_macroarea" : row["macroarea"],
                                    }
                macroareas[row["macroarea"]] = {"id" : row["macroarea"], "entity_type" : "macroarea"}
                families[row["family"]] = {"id" : row["family"], "entity_type" : "family"}
                genuses[row["genus"]] = {"id" : row["genus"], 
                                         #"genus_to_family" : row["family"], 
                                         "entity_type" : "genus"}

    tweets = [] #, languages, families, genuses, macroareas = [], [], [], [], []
    with gzip.open(args.tweet_file, "rt") as ifd:
        for line in ifd:
            tid, lang_code, text = line.strip().split("\t")
            tweet_langs.add(lang_code)
            tweets.append({"id" : tid,
                           "entity_type" : "tweet",
                           "tweet_to_language" : lang_code,
                           "tweet_text" : text,
                           "tweet_language" : lang_code,
            })

    slanguages, sfamilies, sgenuses, smacroareas = set(), set(), set(), set()
    with gzip.open(args.output_file, "wt") as ofd:
        for tweet in tweets:
            if tweet["tweet_to_language"] in wals_langs:
                ofd.write(json.dumps(tweet) + "\n")
                slanguages.add(tweet["tweet_to_language"])
        for code in slanguages:
            lang = languages[code]
            ofd.write(json.dumps(lang) + "\n")
            #sfamilies.add(genuses[lang["language_to_genus"]]["genus_to_family"])
            sgenuses.add(lang["language_to_genus"])
            #smacroareas.add(lang["language_to_macroarea"])
        #for fam in sfamilies:
        #    ofd.write(json.dumps(families[fam]) + "\n")
        for gen in sgenuses:
            ofd.write(json.dumps(genuses[gen]) + "\n")
        #for mac in smacroareas:
        #    ofd.write(json.dumps(macroareas[mac]) + "\n")
