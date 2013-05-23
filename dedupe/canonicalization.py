from distance.affinegap import affineGapDistance
import itertools
from collections import defaultdict
import numpy

values = (("129", "TRK-PAC (Midwest Truckers Assn.)", "", "2727 North Dirksen Parkway", "", "Springfield", "IL", "62702"),
("126338", "TRK-PAC", "", "2727 N. Dirksen", "", "Springfield", "IL", "62702-1490"),
("390787", "TRK PAC", "", "2727 N. Dirksen", "", "Springfield", "IL", "62702"),
("18056", "TRK-PAC", "", "2727 N. Dirkson Parkway", "", "Springfield", "IL", "62702"),
("8202", "TRK-PAC", "", "2727 N Dirksen Parkway", "", "Springfield", "IL", "62702"),
("113039", "TRK-PAC", "", "Mid-West Truckers Assoc.", "2727 N. Dirksen Pkwy.", "Springfield", "IL", "62702"),
("129944", "TRK-PAC", "", "2727 N. Dirksen Pky", "", "Springfield", "IL", "62702-1490"),
("124188", "TRK-PAC Mid-West Truckers Assoc", "", "2727 N. Dirksen", "", "Springfield", "IL", "62702"),
("110115", "TRK-PAK", "", "2727 N. Dirksen Parkway", "", "Springfield", "IL", "62702"),
("62501", "TRK-PAC (Mid-West Truckers Assn.)", "", "2727 N. Dirksen Pkwy", "", "Springfield", "IL", "62702-1490"),
("21618", "TRK-PAC", "", "2727 N. Dirksen Parkway", "", "Springfield", "IL", "62702-1490"),
("123566", "TRK-PAC (Midwest Truckers)", "", "2727 N Dirkson Pkwy", "", "Springfield", "IL", "62702-1490"),
("53680", "TRK PAC", "", "2727 North Dirksen Parkway", "", "Springfield", "IL", "62702"),
("28210", "TRK PAC", "", "2727 N. Dirksen Pkwy.", "", "Springfield", "IL", "62702"),
("390067", "TRK-PAC", "", "Mid-West Truckers Association Inc.", "2727 N. Dirksen Pkwy.", "Springfield", "IL", "62702"),
("21684", "TRK-PAC", "", "2715 N. Dirksen Pkwy.", "", "Springfield", "IL", "62702"),
("21429", "TRK-PAC", "", "2727 N. DIRKSEN PARKWAY", "", "SPRINGFIELD", "IL", "62702"),
("2231", "TRK-PAC", "", "2727 N. Dirksen Pkwy.", "", "Springfield", "IL", "62702-1490"),
("131061", "TRK PAC", "", "2727 N. Dirken Parkway", "", "Springfield", "IL", "62702-1490"),
("52671", "TRK-PAC", "", "2727 N Dirksen Pkwy", "", "Springfield", "IL", "62702-1490"),
("60615", "TRK-PAC", "", "2727 No. Dirksen Parkway", "", "Springfield", "IL", "62702"),
("53577", "TRK-PAC", "", "2727 N. Dirksin Parkway", "", "Springfield", "IL", "62702"),
("118608", "TRK-PAC (Mid-West Truckers Assn.)", "", "2727 N. Dirksen Pkwy", "", "Springfield", "IL", "62702"),
("61405", "TRK-PAC Midwest Trucking Association Inc.", "", "2727 N. Dirkson Parkway", "", "Springfield", "IL", "62702-7410"),
("109407", "TRK-PAC (Mid-West Truckers Assoc. Inc.)", "", "2727 N. Dirksen Pkwy", "", "Springfield", "IL", "62702-1490"),
("11758", "TRK-PAC", "", "2727 N. Dirksen Pkwy", "", "Springfield", "IL", "62702-1490"),
("8303", "TRK-PAC", "", "2727 North Dirksen Parekway", "", "Springfield", "IL", "62702"),
("126192", "TRK-PAC", "", "2727 N. Durksin Parkway", "", "Springfield", "IL", "62702"),
("67570", "TRK-PAC Mid-West Truckers Assoc", "", "2727 N. Dirksen Parkway", "", "Springfield", "IL", "62702"),
("98933", "TRK-PAC/Mid-West Truckers", "", "2727 N Dirksen Pkwy.", "", "Springfield", "IL", "607021490"),
("129401", "TRK PAC", "", "2727 N. Dirksen Pkwy", "", "Springfield", "IL", "62702"),
("80636", "TRK-PAC", "", "2727 N Dirksen Parkway", "", "Springfield", "IL", "62702-1490"),
("125565", "TRK-PAC", "", "2727 N Dirksen Pky", "", "Springfield", "IL", "62702"))

keys = ( "last_name", "first_name", "address_1", "address_2",
        "city", "state", "zip")

cluster = {}
for value in values :
  cluster[value[0]] = dict(zip(keys, value[1:]))


#distance = defaultdict(int)
distance = defaultdict(lambda : defaultdict(int))

for pair in itertools.combinations(cluster.keys(),2) :
  id_1, id_2 = pair
  record_1 = cluster[id_1]
  record_2 = cluster[id_2]
  for field in keys :
    field_1 = record_1[field] if record_1[field] else " "
    field_2 = record_2[field] if record_2[field] else " "
    d = affineGapDistance(record_1[field], record_2[field],
                          matchWeight = 0,
                          mismatchWeight = 1,
                          gapWeight = 1,
                          spaceWeight = 1,
                          abbreviation_scale = 1)
    if numpy.isnan(d) :
      d = 0
    distance[id_1][field] += d
    distance[id_2][field] += d


min_score = dict(zip(keys, [10000000000000] * 7))
print min_score



best_fields = defaultdict(str)

for k, v in distance.items() :
  for field, dis in v.items() :
    if dis < min_score[field] :
      min_score[field] = dis
      best_fields[field] = k

for field in best_fields :
  print field
  print cluster[best_fields[field]][field]
    

print best_fields
