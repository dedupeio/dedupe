import collections
import functools
import numpy
import usaddress
from affinegap import normalizedAffineGapDistance as compareString

STREET_PARTS = (('AddressNumberPrefix',
                 'AddressNumber',
                 'AddressNumberSuffix'),
                ('StreetNamePreDirectional',
                 'StreetNamePostDirectional'),
                ('StreetNamePreModifier',
                 'StreetName',
                 'StreetNamePostModifier'),
                ('StreetNamePostType',
                 'StreetNamePreType'),
                ('OccupancyType',),
                ('OccupancyIdentifier',),
                ('BuildingName',))

BOX_PARTS =  (('USPSBoxGroupType',),
              ('USPSBoxGroupID',),
              ('USPSBoxType',),
              ('USPSBoxID',))

def consolidateAddress(address, components) :
    for component in components :
        merged_component = ' '.join(address[part]  
                                    for part in component
                                    if part in address)
        yield merged_component


def zipAddress(address_1, address_2, components) :
    merged_address_1 = consolidateAddress(address_1, components)
    merged_address_2 = consolidateAddress(address_2, components)
    for component_1, component_2 in zip(merged_address_1, merged_address_2) :
        yield component_1, component_2

def compareFields(address_1, address_2, parts) :
    for part_1, part_2 in zipAddress(address_1, 
                                     address_2,
                                     parts) :
        yield compareString(part_1, part_2)

def compareIntersections(address_1, address_2) :
    street_1 = (('StreetNamePreDirectional',
                 'StreetNamePostDirectional'),
                ('StreetNamePreModifier',
                 'StreetName',
                 'StreetNamePostModifier'),
                ('StreetNamePostType',
                 'StreetNamePreType'))
    street_2 = (('SecondStreetNamePreDirectional',
                 'SecondStreetNamePostDirectional'),
                ('SecondStreetNamePreModifier',
                 'SecondStreetName',
                 'SecondStreetNamePostModifier'),
                ('SecondStreetNamePostType',
                 'SecondStreetNamePreType'))

    address_1A = tuple(consolidateAddress(address_1, street_1))
    address_1B = tuple(consolidateAddress(address_1, street_2))
    address_2 = tuple(consolidateAddress(address_2, street_1 + street_2))

    print address_1A
    print address_1B
    print address_2
    unpermuted_distances = []
    for part_1, part_2 in zip(address_1A + address_1B, address_2) :
        unpermuted_distances.append(compareString(part_1, part_2))

    permuted_distances = []
    for part_1, part_2 in zip(address_1B + address_1A, address_2) :
        permuted_distances.append(compareString(part_1, part_2))

    if numpy.nansum(permuted_distances) < numpy.nansum(unpermuted_distances) :
        return permuted_distances
    else :
        return unpermuted_distances
    
class USAddressType(object) :
    
    AddressType = collections.namedtuple('AddressType', 
                                         ['compare', 'indicator', 
                                          'size', 'offset'])

    
    components = {'Street Address' :
                      AddressType(compare=functools.partial(compareFields,
                                    parts = STREET_PARTS),
                                  size = len(STREET_PARTS),
                                  indicator=[1, 0, 0],
                                  offset=0),
                  'PO Box' :
                      AddressType(compare=functools.partial(compareFields,
                                        parts = BOX_PARTS),
                                  size = len(BOX_PARTS),
                                  indicator=[0, 1, 0],
                                  offset= 2 * len(STREET_PARTS)),
                  'Intersection' :
                      AddressType(compare=compareIntersections,
                                  size=6,
                                  indicator=[0,0,1],
                                  offset = 2 * (len(STREET_PARTS) 
                                                + len(BOX_PARTS)))}

    
    
    def __init__(self) :

        # missing? + same_type? + len(indicator) + ...
        self.expanded_size = 1 + 1 + 3 + 2 * sum(address_type.size
                                                 for address_type 
                                                 in self.components.values())


    def comparator(self, field_1, field_2) :
        distances = numpy.zeros(self.expanded_size)
        i = 0

        if not (field_1 and field_2) :
            return distances
        
        distances[0] = 1
        i += 1

        address_1, address_type_1 = usaddress.tag(field_1) 
        address_2, address_type_2  = usaddress.tag(field_2)

        if address_type_1 != address_type_2 :
            return distances
        
        distances[1] = 1
        i += 1

        compare, address_type, size, offset = self.components[address_type_1]

        distances[2:5] = address_type
        i += 3

        start = i + offset

        for j, dist in enumerate(compare(address_1, address_2), start) :
            distances[j] = dist

        i = j + 1

        print distances
        unobserved_parts = numpy.isnan(distances[start:i])
        distances[start:i][unobserved_parts] = 0
        distances[i:(i + size)] = (~unobserved_parts).astype(int)

        return distances


        
