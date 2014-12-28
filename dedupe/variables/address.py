from dedupe.fieldclasses import DerivedType, StringType
import collections
import functools
import numpy
import usaddress
from affinegap import normalizedAffineGapDistance as compareString

STREET = (('address number',   ('AddressNumberPrefix',
                                'AddressNumber',
                                'AddressNumberSuffix')),
          ('street direction', ('StreetNamePreDirectional',
                                'StreetNamePostDirectional')),
          ('street name',      ('StreetNamePreModifier',
                                'StreetName',
                                'StreetNamePostModifier')),
          ('street type',      ('StreetNamePostType',
                                'StreetNamePreType')),
          ('occupancy type',   ('OccupancyType',)),
          ('occupancy id',     ('OccupancyIdentifier',)),
          ('building name',    ('BuildingName',)))

BOX =  (('box group type', ('USPSBoxGroupType',)),
        ('box group id',   ('USPSBoxGroupID',)),
        ('box type',       ('USPSBoxType',)),
        ('box id',         ('USPSBoxID',)))

INTERSECTION_A = (('street direction A', ('StreetNamePreDirectional',
                                          'StreetNamePostDirectional')),
                  ('street name A',      ('StreetNamePreModifier',
                                          'StreetName',
                                          'StreetNamePostModifier')),
                  ('street type A',      ('StreetNamePostType',
                                          'StreetNamePreType')))
INTERSECTION_B = (('street direction B', ('SecondStreetNamePreDirectional',
                                          'SecondStreetNamePostDirectional')),
                  ('street name B',      ('SecondStreetNamePreModifier',
                                          'SecondStreetName',
                                          'SecondStreetNamePostModifier')),
                  ('street type B',      ('SecondStreetNamePostType',
                                          'SecondStreetNamePreType')))
STREET_NAMES, STREET_PARTS = zip(*STREET)
BOX_NAMES, BOX_PARTS = zip(*BOX)
INTERSECTION_A_NAMES, INTERSECTION_A_PARTS = zip(*INTERSECTION_A)
INTERSECTION_B_NAMES, INTERSECTION_B_PARTS = zip(*INTERSECTION_B)

AddressType = collections.namedtuple('AddressType', 
                                     ['compare', 'indicator', 
                                      'offset'])

def consolidate(d, components) :
    for component in components :
        merged_component = ' '.join(d[part]  
                                    for part in component
                                    if part in d)
        yield merged_component

def compareFields(address_1, address_2, parts) :
    joinParts = functools.partial(consolidate, components=parts)    
    for part_1, part_2 in zip(*map(joinParts, [address_1, address_2])) :
        yield compareString(part_1, part_2)

def compareIntersections(address_1, address_2) :
    street_1 = INTERSECTION_A_PARTS
    street_2 = INTERSECTION_B_PARTS

    address_1A = tuple(consolidate(address_1, street_1))
    address_1B = tuple(consolidate(address_1, street_2))
    address_2 = tuple(consolidate(address_2, street_1 + street_2))

    unpermuted_distances = [compareString(part_1, part_2)
                            for part_1, part_2 
                            in zip(address_1A + address_1B, address_2)]

    permuted_distances = [compareString(part_1, part_2)
                          for part_1, part_2 
                          in zip(address_1B + address_1A, address_2)]

    if numpy.nansum(permuted_distances) < numpy.nansum(unpermuted_distances) :
        return permuted_distances
    else :
        return unpermuted_distances


class USAddressType(StringType) :
    type = "USAddress"

    components = {'Street Address' :
                      AddressType(compare=functools.partial(compareFields,
                                            parts = STREET_PARTS),
                                  indicator=[1, 0, 0],
                                  offset=0),
                  'PO Box' :
                      AddressType(compare=functools.partial(compareFields,
                                            parts = BOX_PARTS),
                                  indicator=[0, 1, 0],
                                  offset= len(STREET)),
                  'Intersection' :
                      AddressType(compare=compareIntersections,
                                  indicator=[0, 0, 1],
                                  offset = len(STREET 
                                               + BOX))}

    # missing? + same_type? + len(indicator) + ...
    expanded_size = 1 + 1 + 3 + 2 * len(STREET 
                                        + BOX
                                        + INTERSECTION_A
                                        + INTERSECTION_B)

    def __len__(self) :
        return self.expanded_size


    def __init__(self, definition) :
        super(USAddressType, self).__init__(definition)

        preamble = [('%s: Not Missing' % definition['field'], 'Dummy'),
                    ('same address type?', 'Dummy'),
                    ('street address Type', 'Dummy'),
                    ('po box', 'Dummy'),
                    ('intersection', 'Dummy')]

        address_parts = [(part, 'String') 
                         for part
                         in (STREET_NAMES 
                             + BOX_NAMES 
                             + INTERSECTION_A_NAMES 
                             + INTERSECTION_B_NAMES)]

        self.n_parts = len(address_parts)

        not_missing_address_parts = [('%s: Not Missing' % part, 'Not Missing') 
                                     for part, _ in address_parts]

        fields = preamble + address_parts + not_missing_address_parts
        
        self.higher_vars = [DerivedType({'name' : name,
                                         'type' : field_type})
                            for name, field_type in fields]

    def comparator(self, field_1, field_2) :
        distances = numpy.zeros(self.expanded_size)
        i = 0

        if not (field_1 and field_2) :
            return distances
        
        distances[i] = 1
        i += 1

        try :
            address_1, address_type_1 = usaddress.tag(field_1) 
            address_2, address_type_2  = usaddress.tag(field_2)
        except Exception as e :
            print e
            return distances

        if address_type_1 != address_type_2 :
            return distances
        
        distances[i] = 1
        i += 1

        address_type = self.components[address_type_1]

        distances[i:5] = address_type.indicator
        i += 3

        i += address_type.offset
        for j, dist in enumerate(address_type.compare(address_1, address_2), 
                                 i) :
            distances[j] = dist

        unobserved_parts = numpy.isnan(distances[i:j+1])
        distances[i:j+1][unobserved_parts] = 0
        unobserved_parts = (~unobserved_parts).astype(int)
        distances[(i + self.n_parts):(j + 1 + self.n_parts)] = unobserved_parts

        return distances


        
