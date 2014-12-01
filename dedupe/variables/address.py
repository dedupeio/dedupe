import collections
import functools
import numpy
import usaddress
from affinegap import normalizedAffineGapDistance as compareString




def consolidateAddress(address, components) :
    for component in components :
        merged_component = ' '.join(address.get(part, '') 
                                    for part in component)
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
    
    # Don't include Recipient, Place, State, Country, Zip
    #
    #
    # needs to do handle missing for entire field, in addition to missing
    # for individual components
    
    # need to handle PO box vs street address
    # 

    # Address Type (Street, PO Box, Intersection) : Exact with Missing
    # 
    # Street Address
    # - Street Dir (pre and post)
    # - Street Type (pre and post)
    # - Addr # (Address Number and Prefix and Suffix)
    # - Unit Type
    # - Unit Number
    # - Street Name (Pre and Post Modifier)
    # - Building Name
    # - Subaddress 
    # - Subaddress ID
    #
    # Intersection
    # - Street Dir1 (pre and post)
    # - Street Type1 (pre and post)
    # - Street Name1 (Pre and Post Modifier)
    # - Street Dir2 (pre and post)
    # - Street Type2 (pre and post)
    # - Street Name2 (Pre and Post Modifier)
    #
    # Compare 
    # [(field1, Street1) and (field2, Street1), 
    #  (field1, Street2) and (field2, Street2)] 
    # AND
    # [(field1, Street2) and (field2, Street1), 
    #  (field1, Street1) and (field2, Street2)] 
    # Use min distance
    #
    # PO Box
    # - Box Type
    # - Box ID
    # - BoxGroup Type
    # - BoxGroup ID
    #
    AddressType = collections.namedtuple('AddressType', 
                                         ['compare', 'indicator', 
                                          'size', 'offset'])

    
    components = {'Street Address' :
                      AddressType(compare=functools.partial(compareFields,
                                        parts = (('AddressNumberPrefix',
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
                                                 ('BuildingName',))),
                                  size = 7,
                                  indicator=[1, 0, 0],
                                  offset=0),
                  'PO Box' :
                      AddressType(compare=functools.partial(compareFields,
                                        parts = (('USPSBoxGroupType',),
                                                 ('USPSBoxGroupID',),
                                                 ('USPSBoxType',),
                                                 ('USPSBoxID',))),
                                  size = 4,
                                  indicator=[0, 1, 0],
                                  offset=14),
                  'Intersection' :
                      AddressType(compare=compareIntersections,
                                  size=6,
                                  indicator=[0,0,1],
                                  offset=22)}

    
    
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

        compare, indicator, size, offset = self.components[address_type_1]

        distances[2:5] = indicator
        i += 3

        start = i + offset

        for j, dist in enumerate(compare(address_1, address_2), start) :
            distances[j] = dist

        print distances

        i = j + 1

        unobserved_parts = numpy.isnan(distances[start:i])
        distances[start:i][unobserved_parts] = 0
        distances[i:(i + size)] = (~unobserved_parts).astype(int)

        return distances


        # ['not addr', 'Null'],
        # ['addr #', 'AddressNumber'],
        # ['st dir pre', 'StreetNamePreDirectional'],
        # ['st dir post', 'StreetNamePostDirectional'],
        # ['st name', 'StreetName'],
        # ['st type post', 'StreetNamePostType'],
        # ['st type pre', 'StreetNamePreType'],
        # ['intersection separator', 'IntersectionSeparator'],
        # ['unit type', 'OccupancyType'],
        # ['unit no', 'OccupancyIdentifier'],
        # ['box type', 'USPSBoxType'],
        # ['box no', 'USPSBoxID'],
        # ['city', 'PlaceName'],
        # ['state', 'StateName'],
        # ['zip', 'ZipCode'],
        # ['zip+4', 'ZipPlus4'],
        # ['country', 'CountryName'],
        # ['landmark', 'LandmarkName'],
        # ['box type', 'USPSBoxType'],
        # ['box no', 'USPSBoxID'],

        # ['box group type', 'USPSBoxGroupType'],
        # ['box group id', 'USPSBoxGroupID'],
        # ['address number prefix', 'AddressNumberPrefix'],
        # ['address number suffix', 'AddressNumberSuffix'],
        # ['subaddress id', 'SubaddressIdentifier'],
        # ['subaddress type', 'SubaddressType'],
        # ['recipient', 'Recipient'],
        # ['streetname modifer, pre', 'StreetNamePreModifier'],
        # ['streetname modifer, post', 'StreetNamePostModifier'],
        # ['building name', 'BuildingName'],
        # ['corner/junction', 'CornerOf']


        
