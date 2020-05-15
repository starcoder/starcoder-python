import argparse
import gzip
import pickle
from glob import glob
import os.path
from data import Dataset, Spec
import logging
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-s", "--spec_output", dest="spec_output", help="Output file")
    parser.add_argument("-d", "--data_output", dest="data_output", help="Output file")
    parser.add_argument("-c", "--collapse", dest="collapse", default=False, action="store_true", 
                        help="Move language distributions from tweets to users")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    field_types = dict([
        ("distribution", ["valid_languages", "fluency_languages"] + (["twitter_languages"] if args.collapse else [])),
        ] + ([("categorical", [None])] if args.collapse else [("categorical", [None, "twitter_language"])]))

    entity_type_to_fields = dict([
        ("user", set(["fluency_languages"] + (["twitter_languages", "valid_languages"] if args.collapse else []))),
    ] + ([] if args.collapse else [("tweet", set(["twitter_language", "valid_languages"]))]))

                                 
    entity_relation_types = dict([
        ("follows", ("user", "user")),
        ("followed_by", ("user", "user")),
    ] + ([] if args.collapse else [("tweeted", ("user", "tweet")), 
                                   ("tweeted_by", ("tweet", "user")),
                                   ]))
                                 
    field_values = {
        "fluency_languages" : set(["und", "pol", "lav", "eng", "bul", "rus", "ukr"]),
        "twitter_language" : set("ita tur pan ice fre urd pol lav ckb iku por tib tha ori tgl kan und bur chr bul khm in hin mar ger slo wel tel lit geo dan srp arm sin tam slv iw uig hrv mal est ukr per eng snd rus amh hun rum hat nep fin nor gre kor ben ara guj swe chi vie dut spa pus jpn bos lao".split()),
        "twitter_languages" : set("ita tur pan ice fre urd pol lav ckb iku por tib tha ori tgl kan und bur chr bul khm in hin mar ger slo wel tel lit geo dan srp arm sin tam slv iw uig hrv mal est ukr per eng snd rus amh hun rum hat nep fin nor gre kor ben ara guj swe chi vie dut spa pus jpn bos lao".split()),
        "valid_languages" : set("hy glk ug mr li ak bm su os pnt diq ay sn lij vi mt ca nv gan uk gl ta oc ko ts lmo mrj mzn ba mg ar es gv haw cdo br ti cs bi ltg et war el crh lb kaa zu si bn als be no ki zea ilo sa tt id udm rue sv ff hr lt fur rn fr ceb as pam kl yo hu chr ik ang bg frr eu ckb hak arz pag fa so gag tn mwl hi bcl zh scn za ve kw sq pnb mk ms eml te xal ku ce new cy sw pi ps pt gn ab it ur ace nov co qu en sd my pap rmy sm mi lad nap ty mhr xh gd ka sr ch io wa tpi hif kn uz ne bxr sco tl nl ie bpy mo lbe nn krc de kk ny koi gu ml kbd vep jv ro av kab bug nso ks cv hsb rm to fy or ss fi wuu chy xmf sk kg lez fo ast csb mdf frp yi pl rw sl ee he tk an ext pcd bs sg tw lv vec ja pms om ky pa sah iu eo ga am arc nds sh fj kv lg bjn az ru pfl vls got ha af nah bh jbo vo th dz stq se na tum tr is lo ia la bo dsb myv wo tet st mn cu ig szl tg srn ln ht bar cr da pdc dv sc pih km".split()),
        None : set(["user"] + ([] if args.collapse else ["tweet"])),
    }
                                 
    field_types = dict(sum([[(f, t) for f in ff] for t, ff in field_types.items()], []))
    field_values = {k : field_values[k] for k in field_types.keys()}

    spec = Spec({k : v for k, v in field_types.items() if k in field_values}, 
                field_values, 
                entity_type_to_fields,
                entity_relation_types)

    logging.info("Created %s", spec)


    relationships = {}
    id_to_index = {}
    index_to_entity = {}

    for fi, fname in enumerate(args.inputs):
        with gzip.open(fname, "rb") as ifd:
            target, edges, content = pickle.load(ifd)
            content = {(fi, int(k)) : v for k, v in content.items()}
            target_id = (fi, int(target["id"]))
            if target.get("labels", {}).get("bot", False) == False and target_id in content:
                fluency_dist = {l : 1.0 for l in target.get("fluencies", {}).keys()}
                for userid, tweets in content.items():
                    uindex = id_to_index.setdefault(userid, len(id_to_index))
                    index_to_entity[uindex] = index_to_entity.get(uindex, {None : "user"})
                    if userid == target_id:
                        index_to_entity[uindex]["fluency_languages"] = fluency_dist

                    if args.collapse:                        
                        twitter_dist = {}
                        valid_dist = {}
                        for tweetid, tweet in tweets.items():
                            tlang = list(tweet["twitter"][1].keys())[0]
                            twitter_dist[tlang] = twitter_dist.get(tlang, 0.0) + 1.0
                            vlangs = {k : math.exp(v) for k, v in tweet["valid"].items()}                            
                            total = sum(vlangs.values())
                            for k, v in vlangs.items():
                                valid_dist[k] = valid_dist.get(k, 0.0) + (v / total)
                        index_to_entity[uindex]["twitter_languages"] = twitter_dist
                        index_to_entity[uindex]["valid_languages"] = valid_dist
                    else:
                        for tweetid, tweet in tweets.items():
                            tweetid = (fi, int(tweetid))
                            tindex = id_to_index.setdefault(tweetid, len(id_to_index))
                            index_to_entity[tindex] = index_to_entity.get(tindex, {None : "tweet"})
                            relationships["tweeted"] = relationships.get("tweeted", [])
                            relationships["tweeted"].append((uindex, tindex))
                            relationships["tweeted_by"] = relationships.get("tweeted_by", [])
                            relationships["tweeted_by"].append((tindex, uindex))
                for s, t in edges:
                    s = (fi, int(s))
                    t = (fi, int(t))
                    if s in id_to_index and t in id_to_index:
                        relationships["follows"] = relationships.get("follows", [])
                        relationships["follows"].append((id_to_index[s], id_to_index[t]))
                        relationships["followed_by"] = relationships.get("followed_by", [])
                        relationships["followed_by"].append((id_to_index[t], id_to_index[s]))
                
            logging.info("Currently at %d total entities after reading '%s'", len(index_to_entity), fname)

    data = Dataset(spec, [index_to_entity[i] for i in range(len(index_to_entity))], relationships)

    with gzip.open(args.spec_output, "wb") as ofd:
        pickle.dump(spec, ofd)
    
    with gzip.open(args.data_output, "wb") as ofd:
        pickle.dump(data, ofd)
        
    logging.info("Created %s, wrote to %s and %s", data, args.spec_output, args.data_output)
