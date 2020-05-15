import argparse
import gzip
import json
import re
import sys
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import calendar
import pickle

months = [x.lower() for x in list(calendar.month_name)]

def fix_date(*args):
    args = list(args)
    year, month, day = False, False, False
    nans = len([x for x in args if isinstance(x, (float, int)) and numpy.isnan(x)])
    if args in [("Year", "Month", "Day"), ("Event Yr.", "Event Mo.", "Event Day")] or nans >= 2:
        #bad_dates.append(args)
        #good_dates.append(args)
        return None
    elif len(args) == 1:
        toks = args[0].strip().strip("?").replace("`", "").strip().split()
        if len(toks) == 3:
            day = int(toks[0])
            month = toks[1]
            year = int(toks[2])
        elif len(toks) == 2:
            month = toks[0]
            try:
                year = int(toks[1])
            except:
                return None
            day = 1
    elif len(args) == 3 and isinstance(args[1], str): # and isinstance(args[2], (int, float)):
        try:
            args[0] = float(args[0])
        except:
            return None
        if args[0] < 1000:
            return None
        try:
            year = int(args[0])
        except:
            return None
        year = 1827 if year == 11827 else year
        month = args[1]
        day = int(float(1 if args[2] in ["Charlt", "4S", "woman named Rhino"] or args[2] == "nan" else args[2]))
    if day and month and year:
        month = month.replace("Janurary", "January").replace("Septemer", "September").replace("Ocotber", "October").replace("Arpril", "April").replace("Dececember", "December").replace("Novemer", "November").replace("Juliy", "July").replace("Juy", "July").strip().rstrip("?").strip().replace("Janauary", "January")
        month = "February" if month == "Feb" or month == "f" else month
        month = "May" if month == "Ma" else month
        try:
            month_id = months.index(month.lower())
        except:
            return None
        if month_id == 0:
            return None
        day = int(30 if day > 31 else day)
        #good_dates.append(args)
        return "{}-{:0>2}-{:0>2}".format(year, month_id, day)


def fix_name(*args):
    if len(args) == 1:
        name_toks = args[0].lower().split()
        if len(name_toks) == 0 or "not listed" in args[0]:
            return None
        elif len(name_toks) == 1:
            return {"last" : name_toks[0].title()}
        else:
            return {"first" : name_toks[0].title(),
                    "last" : " ".join(name_toks[1:]).title(),
                    "full" : " ".join(name_toks).title()}
        
    elif len(args) == 2:
        if args[1] in ["nan"]:
            return None
        elif args[0] in ["nan"]:
            return {"last" : args[1].title()}
        else:
            return {"first" : args[0].title(),
                    "last" : args[1].title(),
                    "full" : "{} {}".format(args[0], args[1]).title(),
                    }
            
    else:
        return None

def fix_slave_name(*args):
    args = [x for x in args if x != "nan"]
    if len(args) == 1:
        name_toks = args[0].lower().split()
        if len(name_toks) == 0 or "not listed" in args[0]:
            return None
    elif len(args) == 2:
        name_toks = args
    else:
        return None
    if len(name_toks) == 1 and name_toks != "nan":
        return {"first" : name_toks[0].title()}
                #"full" : name_toks[0].title()}
    else:
        return {"first" : name_toks[0].title(),
                "last" : " ".join(name_toks[1:]).title(),
                "full" : " ".join(name_toks).title()}

    
class Entities(dict):
    def __init__(self, schema):
        self._key_to_id = {}
        self._schema = {k : v[0] for k, v in schema.items()}
        self._aggregates = {k : v[1] for k, v in schema.items()}
        for k in self._schema.keys():
            self[k] = {}
        self._totals = {k : 0 for k in schema.keys()}
        self._total = 0
        
    def get_id(self, rtype, record):
        key = tuple(["{}".format(record.get(k, "")) for k in self._schema[rtype]])
        #key = tuple(["{}".format(record[k]) for k in self._schema[rtype]])
        return self._key_to_id[(rtype, key)]
    
    def merge(self, record):
        rtype = record["type"]
        self._totals[rtype] += 1
        record = {k : v for k, v in record.items() if v not in ["nan", None, ""]}
        key = [record.get(k, None) for k in self._schema[rtype]]
        if all([x == None for x in key]):
            return None
        key = tuple(["{}".format(record.get(k, "")) for k in self._schema[rtype]])
        if key not in self[rtype]:
            self[rtype][key] = record
            for k in record.keys():
                if k in self._aggregates[rtype]:
                    self[rtype][key][k] = [self[rtype][key][k]]
            self._key_to_id[(rtype, key)] = len(self._key_to_id)

        else:
            for k, v in record.items():
                if k not in self[rtype][key]:
                    if k in self._aggregates[rtype]:
                        if k not in self[rtype][key]:
                            self[rtype][key][k] = []
                        self[rtype][key][k].append(v)
                    else:
                        self[rtype][key][k] = v
                else:
                    pass
        return self._key_to_id[(rtype, key)]
            
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument("-H", "--host", dest="host", default="localhost", help="Host")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9200, help="Port")
    parser.add_argument("-s", "--source_index_name", dest="source_index_name", default="original_slavery", help="Source index name")
    parser.add_argument("-t", "--target_index_name", dest="target_index_name", default="slavery", help="Target index name")
    parser.add_argument("-l", "--locations", dest="locations", help="Locations file")
    parser.add_argument("-d", "--dry_run", dest="dry_run", default=False, action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    entities = Entities({"subscriber" : (["subscriber_name"], []),
                         "owner" : (["owner_name"], []),
                         "vessel" : (["vessel_name"], []),
                         "consigner" : (["consigner_name"], []),
                         "captain" : (["captain_name"], []),
                         "shipper" : (["shipper_name"], []),
                         "slave" : (["slave_name"], []),
                         "prisoner" : (["slave_name"], []),
                         "jail" : (["jail_name"], []),
                         "voyage" : (["vessel_name", "manifest_date"], ["slave_name"]),
                         "notice" : (["gazette", "owner_name", "notice_date"], ["slave_name"]),
                         "location" : ([], []),
                         "date" : ([], []),
                         "gazette" : ([], []),
    })
    # aggregates
    

    locations = {}
    with open(args.locations, "rb") as ifd:
        for loc_string, loc in pickle.load(ifd).items():
            if len(loc["features"]) > 0:
                feat = loc["features"][0]["properties"]
                locations[loc_string] = [feat["lng"], feat["lat"]]
                
    logging.info("Loaded %d locations", len(locations))
    items = []
    uvals = {}
    if args.input:
        with gzip.open(args.input, "rt") as ifd:
            for line in ifd:
                items.append(json.loads(line))
    else:
        es = Elasticsearch([{"host" : args.host, "port" :  args.port}])
        while True:
            res = es.search(index=args.source_index_name, body={"query": {"match_all": {}}}, from_=len(items), size=10000)
            if len(res["hits"]["hits"]) == 0:
                break
            items += [h["_source"] for h in res["hits"]["hits"] if h["_source"].get("Recent Punishment") != "Recent Punishment"]

            
    logging.info("Pulled %d original records", len(items))
    for item in items:
        for k, v in item.items():
            uvals[k] = uvals.get(k, set())
            uvals[k].add(v)

    
    boolean_fields = ["is_migration"]
    keyword_fields = ["vessel_type", "source", "gender", "race", "type", "ad_venue", "type"]
    text_fields = ["notes", "alias", "string"]
    integer_fields = ["row", "captives"]
    float_fields = ["reward", "age"] #, "tonnage"]
    time_fields = ["notice_date", "event_date", "birth_date", "event_date", "manifest_date", "departure_date", "arrival_date"]
    location_fields = ["residence"] #, "capture_location", "jail_location", "home_port", "arrival_port", "departure_port"]
    parent_fields = ["captain_id", "consignor_id", "vessel_id", "owner_id", "shipper_id", "subscriber_id", "slave_id"]
    child_fields = []
    
    properties = {}
    for f in boolean_fields:
        properties[f] = {"type" : "boolean"}
    for f in keyword_fields:
        properties[f] = {"type" : "keyword"}
    for f in text_fields:
        properties[f] = {"type" : "text"}
    for f in integer_fields:
        properties[f] = {"type" : "integer"}
    for f in float_fields:
        properties[f] = {"type" : "float"}
    for f in time_fields:
        properties[f] = {"type" : "date"} #, "format" : "E MMM d HH:mm:ss Z YYYY"}
    for f in location_fields:
        properties[f] = {"type" : "geo_point"}
    properties["edge"] = {"type" : "join",
                          "relations" : {
                              "owner" : "slave",
                          }
    }
    properties["name"] = {"properties" : {"first" : {"type" : "text"}, "last" : {"type" : "text"}}}
    #properties = {k : v for k, v in properties.items() if k in}
    #print(properties)
    #sys.exit()

    
    # derived_from
    for item in items:
        # Family
        src = (item["Source file"], item["Source sheet"])
        if src == ("fugitive.xlsx", "Sheet1"):
            # owner = Residence, Owner County/City/State/Country
            # capture = Location of Capture
            owner_loc = ", ".join([x for x in [item["Residence"], item["Owner City"], item["Owner County"], item["Owner State"], item["Owner Country"]] if isinstance(x, str) and not re.match(r"^\s*$", x) and x != "nan"])
            #print(owner_loc)
            jail = None
            if item["Type"].strip().lower() == "slave":
                location = {"type" : "location",
                            "description" : owner_loc,
                            "coordinates" : locations.get(owner_loc, None),
                }
                lid = entities.merge(location)                
                owner = {"type" : "owner",
                         "location_id" : lid,
                         "name" : fix_name(item["OWNER FN"], item["OWNER LN"]),
                         "gender" : item["Owner Sex"],
                         #"edge" : {"name" : "owner"},
                }
                oid = entities.merge(owner)                
                fugitive = {"type" : "slave",
                            "name" : fix_name(item["FN"], item["LN"]),
                            "gender" : item["Gender"],
                            "age" : item["Age"],
                            "owner_id" : oid,
                            
                            #"owner_name" : fix_name(item["OWNER FN"], item["OWNER LN"]),
                            # birthday
                            #"edge" : None if oid == None else {"name" : "slave", "parent" : oid},
                }
                fid = entities.merge(fugitive)
                
            elif "jail" in item["Type"].strip().lower():
                jail = {"type" : "jail",
                        "jail_name" : item["Jail"],
                        "jail_state" : item["Jail State"],
                        # location : ...
                }
                jid = entities.merge(owner)
                fugitive = {"type" : "prisoner",
                            "slave_name" : fix_slave_name(item["FN"], item["LN"]),
                            "gender" : item["Gender"],
                            "age" : item["Age"],
                            "jail_id" : jid,
                            # birthday
                }
                fid = entities.merge(fugitive)
            else:
                continue
            
            subscriber = {"type" : "subscriber",
                          "name" : fix_name(item["SUBSCRIBER/AUTHOR FN"], item["SUBSCRIBER/AUTHOR LN"]),
                          "subscriber_type" : item["subscriber/author description"],
            }
            sid = entities.merge(subscriber)

            gazette = {"type" : "gazette",
                       "name" : item["Source"],
            }
            gid = entities.merge(gazette)

            notice_date = {"type" : "date",
                           "notice_date" : fix_date(item["Ad Year"],
                                                    item["Ad Month"],
                                                    item["Ad Day"]),
            }
            nid = entities.merge(notice_date)

            event_date = {"type" : "date",
                          "value" : fix_date(item["Event Yr_"],
                                             item["Event Mo_"],
                                             item["Event Day"])
            }
            eid = entities.merge(event_date)
            
            notice = {"type" : "notice",
                      "gazette_id" : gid,
                      "notice_date_id" : nid,
                      "event_date_id" : eid,
                      "owner_id" : oid,
                      "fugitive_id" : fid,
                      #"owner_name" : fix_name(item["OWNER FN"], item["OWNER LN"]),
                      #"capture_location" : item["Location of Capture"],
                      #"fugitive_name" : fugitive["slave_name"],
            }
            entities.merge(notice)
            #print(notice["notice_date"], item)
            #sys.exit()
            #[entities.merge(e) for e in [jail, fugitive, subscriber, notice] if e != None]
            # Notice, Gazette, Event, Advertiser, Slave, Owner, Subscriber, Location
            
            pass
        elif src == ("norfolk.xlsx", "Master Norfolk"):
            #locs += [row[x] for x in ["OWNER LOC", "SHIPPER LOC", "CONSIGN LOC", "VESSEL HOME PORT", "arrival port"] if not re.match(r"^\s*$", x)]
            # DEPARTURE
            # OWNER LOC
            # SHIPPER LOC
            # CONSIGN LOC
            # VESSEL HOME PORT
            # arrival port
            pass
            # entities.merge({"type" : "owner",
            #                 "residence" : locations.get(item["OWNER LOC"], None),
            #                 "name" : {"first" : item["Owner FN"], "last" : item["Owner LN"]},
            #                 "gender" : None,
            #                 "edge" : {"name" : "owner"},
            # })

            
            # # Voyage, Location, Slave, Owner, Shipper, Consigner, Vessel, Captain, Manifest
            # entities.merge({"type" : "vessel",
            #                 "vessel_name" : item["VESSEL NAME"],
            #                 "vessel_type" : item["VESSEL"],
            #                 "home_port" : locations.get(item["VESSEL HOME PORT"], None),
            #                 "tonnage" : item["TONNAGE"],
            # })

            
        elif src == ("baltimore.xlsx", "Manifest Data"):
            # Owner Location
            # Shipper Location
            # Consignor Location
            #locs += [row[x] for x in ["Owner Location", "Shipper Location", "Consignor Location"] if not re.match(r"^\s*$", x)]            
            # Slave, Owner, Shipper, Consigner, Location, Vessel, Captain, Manifest
            owner = {"type" : "owner",
                     "residence" : locations.get(item["Owner Location"], None),
                     "owner_name" : fix_name(item["Owner"]),
                     "edge" : {"name" : "owner"},                     
            }
            oid = entities.merge(owner)

            slave = {"type" : "slave",
                     "slave_name" : fix_slave_name(item["First name"], item["Last Name"]),
                     #"owner_name" : owner["name"],
                     "edge" : None if oid == None else {"name" : "slave", "parent" : oid},
            }
            
            shipper = {"type" : "shipper",
                       "residence" : locations.get(item["Shipper Location"], None),
                       "shipper_name" : fix_name(item["Shipper "]),
            }

            consigner = {"type" : "consigner",
                         "residence" : locations.get(item["Consignor Location"], None),
                         "consigner_name" : fix_name(item["Consignor"]),
            }

            captain = {"type" : "captain",
                       "captain_name" : fix_name(item["Captain"]),
            }
            
            vessel = {"type" : "vessel",
                      "vessel_name" : item["Vessel"],
                      "vessel_type" : item["Vessel Type"],
                      "tonnage" : item["Tonnage"],
            }
            
            voyage = {"type" : "voyage",
                      "vessel_name" : vessel["vessel_name"],
                      "manifest_date" : fix_date(item["Manifest Date"]),
                      "slave_name" : slave["slave_name"],
                      "captain_name" : captain["captain_name"],
                      "owner_name" : owner["owner_name"],
                      "consigner_name" : consigner["consigner_name"],
                      "shipper_name" : shipper["shipper_name"],
            }
            [entities.merge(e) for e in [vessel, voyage, slave, consigner, captain, shipper]]
            
        elif src == ("baltimore.xlsx", "Voyages"):
            # VESSEL HOME
            # ARRIVAL PORT
            #locs += [row[x] for x in ["VESSEL HOME", "ARRIVAL PORT"] if not re.match(r"^\s*$", x)]            
            # Vessel, Captain, Manifest, Voyage, Location
            captain = {"type" : "captain",
                       "captain_name" : {"first" : item["CAPTAIN"], "last" : item["CAPTAIN_1"]},
            }
            vessel = {"type" : "vessel",
                      "vessel_name" : item["VESSEL NAME"],
                      "vessel_type" : item["VESSEL"],
                      "home_port" : locations.get(item["VESSEL HOME"], None),
                      "tonnage" : item["TONNAGE"],
            }
            voyage = {"type" : "voyage",
                      "departure_port" : item["Documented Origin"],
                      "arrival_port" : item["ARRIVAL PORT"],
                      "vessel_name" : vessel["vessel_name"],
                      "manifest_date" : fix_date(item["LATEST MANIFEST DATE_2"],
                                                 item["LATEST MANIFEST DATE_1"],
                                                 item["LATEST MANIFEST DATE"]),
                      "captain_name" : captain["captain_name"],
                      #"owner_name" : owner["name"],
                      #"consigner_name" : consigner["name"],
                      #"shipper_name" : shipper["name"],
            }
            [entities.merge(e) for e in [vessel, captain, voyage]]            
            pass
            

        else:
            raise Exception(str(src))

    if args.output:
        with gzip.open(args.output, "wt") as ofd:            
            for k, vs in entities.items():
                for i, v in vs.items():
                    ofd.write(json.dumps(v) + "\n")
    else:

        if not args.dry_run:
            if es.indices.exists(args.target_index_name):
                es.indices.delete(index=args.target_index_name)            

            es.indices.create(index=args.target_index_name, body={
                "mappings" : {
                    "properties" : properties,
                }
            })
            es.indices.put_settings(index=args.target_index_name, body={"index" : { "max_result_window" : 500000 }})

        actions = []
        for t, vs in entities.items():
            logging.info("Extracted %d distinct %ss from %d occurrences", len(vs), t, entities._totals[t])
            for i, v in vs.items():
                v["type"] = t
                r = v["edge"]["parent"] if t == "slave" and "edge" in v else entities.get_id(t, v)

                #print(r)
                actions.append({"_source" : v,
                                "_index" : args.target_index_name,
                                "_id" : entities.get_id(t, v),
                                "_routing" : r,
                                #v.get("edge", {}).get("parent", v.get("name"))
                })

        # for k, v in uvals.items():
        #     v = set([x for x in v if x != "nan"])
        #     if len(v) > 100:            
        #         print(k, len(v))
        #     elif len(v) > 0:
        #         print(k, v)

        if not args.dry_run:
            bulk(index=args.target_index_name, actions=actions, raise_on_error=True, client=es)
