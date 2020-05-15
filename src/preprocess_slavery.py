import argparse
import logging
import pandas
import os.path
import json
import gzip
import re
import datetime
import calendar

months = [x.lower() for x in list(calendar.month_name)]
def format_date(val):
    df = "{}_day".format(d)
    ddf = "{}_date".format(d)
    mf = "{}_month".format(d)
    yf = "{}_year".format(d)
    if ddf in src:
        toks = src[ddf].split()
        #del src[ddf]
        if len(toks) == 3:
            day, month, year = toks
            day = int(day)
            if day > 31:
                day = 28
            month = src[mf]
            year = int(float(src[yf]))
            #try:
            src[ddf] = datetime.date(year, months.index(month.lower()), day).toordinal()
        elif len(toks) == 2:
            month, year = toks
            day = 1
            try:
                year = int(float(year))
                src[ddf] = datetime.date(year, months.index(month.lower()), day).toordinal()
            except:
                if ddf in src:
                    del src[ddf]                              
                #except:
            #    print(src)

    elif all([x in src for x in [df, mf, yf]]):
        try:
            day = int(float(src[df]))
            month = src[mf]
            year = int(float(src[yf]))
            date = "{} {} {}".format(year, month, day)
            src[ddf] = datetime.date(year, months.index(month.lower()), day).toordinal()
            #src[ddf] = src.get(ddf, date)
        except:
            if ddf in src:
                del src[ddf]
    elif ddf in src:
        del src[ddf]    
    for f in [df, mf, yf] + remove:
        if f in src:
            del src[f]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(dest="inputs", nargs="+")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    fields = set()
    entities = []
    for fname in args.inputs:
        logging.info("Reading file '%s'", fname)
        sheets = pandas.read_excel(fname, sheet_name=None)
        for sname in [s for s in sheets.keys() if not s.startswith("DPCache")]:
            logging.info("Processing sheet '%s'", sname)
            sheet = sheets[sname]
            for field in sheet.iloc[0].to_dict().keys():
                if not field.startswith("Unnamed"):
                    fields.add(field.replace(".", "_"))
            entity_types = set([f.split("_")[0] for f in fields])
            for i in range(len(sheet)):
                line_entities = {"source" : {"source_file" : os.path.basename(fname),
                                             "source_sheet" : sname,
                                             "source_row" : i + 1}
                                         }
                for f, v in [(k.replace(".", "_"), v) for k, v in sheet.iloc[i].to_dict().items() if not k.startswith("Unnamed")]:
                    et = f.split("_")[0]
                    line_entities[et] = line_entities.get(et, {})
                    line_entities[et][f] = v
                for j, k in enumerate(line_entities.keys()):
                    line_entities[k]["id"] = len(entities) + j
                for et, e in line_entities.items():
                    e["entity_type"] = et
                    for oet, oe in line_entities.items():
                        if oet != et:
                            e["{}_to_{}".format(et, oet)] = oe["id"]
                    entities.append(e)
    assert(all(["entity_type" in x for x in entities]))
    # vals = {}
    # for item in entities:        
    #     if "owner_first_name" in item:
    #         k = "owner_name"
    #         v = "{} {}".format(item.get("owner_first_name", ""), item.get("owner_last_name", ""))
    #         vals[k] = vals.get(k, {})
    #         vals[k][v] = vals[k].get(v, 0) + 1
    #     for k, v in item.items():
    #         vals[k] = vals.get(k, {})
    #         vals[k][v] = vals[k].get(v, 0) + 1
    # for k, v in vals.items():
    #     if len(v) < 20:
    #         print(k, v)
    #print(len(list(sorted([(a, b) for a, b in vals["owner_name"].items()], key=lambda x : x[1]))))
    #sys.exit()
    properties = {f : {"type" : "text"} for f in fields}
    properties["source_file"] = {"type" : "keyword"}
    properties["source_sheet"] = {"type" : "keyword"}
    properties["source_row"] = {"type" : "integer"}

    dates = ["notice_event", "notice", "voyage_arrival", "voyage_departure", "voyage_manifest"]
    numeric = ["slave_age", "vessel_tonnage", "notice_reward_amount", "voyage_count", "notice_party_size", "owner_count"]
    coll_locs = ["owner_location", "owner_state", "owner_county", "owner_city", "owner_country"]
    remove = [] #"voyage_port_2"]

    with gzip.open(args.output, "wt") as ofd:
        for entity in entities:
            src = entity
            for k in list(src.keys()):
                v = str(src[k])
                if v == "nan":
                    del src[k]
                #if re.match("^\s*not", v) or re.match("^\s*\?\s*$", v) or v == "nan":
                #    del src[k]
                else:
                    src[k] = v.replace("?", "").strip()
            for nf in numeric:
                if nf in src:
                    try:
                        src[nf] = float(src[nf])
                    except:
                        del src[nf]

            if any([l in src for l in coll_locs]):
                vals = []
                for k in coll_locs:
                    if k in src:
                        vals.append(src[k])
                        del src[k]
                src["owner_location"] = " ".join(vals)
            for d in dates:
                df = "{}_day".format(d)
                ddf = "{}_date".format(d)
                mf = "{}_month".format(d)
                yf = "{}_year".format(d)
                if ddf in src:
                    toks = src[ddf].split()
                    #del src[ddf]
                    if len(toks) == 3:
                        day, month, year = toks
                        day = int(day)
                        if day > 31:
                            day = 28
                        month = src[mf].strip()
                        year = int(float(src[yf]))
                        #try:
                        src[ddf] = datetime.date(year, months.index(month.lower()), day).toordinal()
                    elif len(toks) == 2:
                        month, year = toks
                        day = 1
                        try:
                            year = int(float(year))
                            src[ddf] = datetime.date(year, months.index(month.lower()), day).toordinal()
                        except:
                            if ddf in src:
                                del src[ddf]                              
                            #except:
                        #    print(src)

                elif all([x in src for x in [df, mf, yf]]):
                    try:
                        day = int(float(src[df]))
                        month = src[mf]
                        year = int(float(src[yf]))
                        date = "{} {} {}".format(year, month, day)
                        src[ddf] = datetime.date(year, months.index(month.lower()), day).toordinal()
                        #src[ddf] = src.get(ddf, date)
                    except:
                        if ddf in src:
                            del src[ddf]
                elif ddf in src:
                    del src[ddf]    
                for f in [df, mf, yf] + remove:
                    if f in src:
                        del src[f]

            for pt in ["author", "slave", "owner", "shipper", "captain", "consignor"]:
                s = "{}_sex".format(pt)
                if s in src:                        
                    src[s] = src[s].lower()
                    if src[s] not in ["m", "f"]:
                        del src[s]

                fn = "{}_first_name".format(pt)
                ln = "{}_last_name".format(pt)
                if ln in src or fn in src:
                    parts = ([src[fn]] if fn in src else []) + ([src[ln]] if ln in src else [])
                    src["{}_name".format(pt)] = " ".join(parts)
                    if ln in src:
                        del src[ln]
                    if fn in src:
                        del src[fn]

                oc = "{}_owner_count".format(pt)
                if oc in src:
                    try:
                        src[oc] = float(src[oc])
                    except:
                        del src[oc]

                hf = "{}_height_feet".format(pt)
                hi = "{}_height_inches".format(pt)
                if hf in src:

                    feet = float(src[hf])
                    if hi in src:
                        inches = float(src[hi])
                    else:
                        inches = 0.0
                    src["{}_height".format(pt)] = feet + (inches / 12.0)
                    del src[hf]
                if hi in src:
                    del src[hi]
            #src["source_row"] = int(src["source_row"])
            assert ("entity_type" in src), str(src)
            ofd.write(json.dumps(src) + "\n")
