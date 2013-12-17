from dedupe.distance.affinegap import normalizedAffineGapDistance
from dedupe.distance.haversine import compareLatLong
from dedupe.distance.jaccard import compareJaccard
from dedupe.distance.categorical import CategoricalComparator

try:
    from collections import OrderedDict
except ImportError :
    from core import OrderedDict


def initializeDataModel(fields):
    """Initialize a data_model with a field definition"""

    # The data model needs to order the different kinds of variabls in
    # a particular way. First, variables that are directly produced by
    # a comparison function have to go first. Second, dummy variable
    # that will be set from the return value of a comparison
    # function. Third, interaction variables of the first two
    # types. Finally, handle any missing data flags. The reasons for this
    # order has to do with implementation details of core.fieldDistances

    data_model = {}
    data_model['fields'] = OrderedDict()

    interaction_terms = OrderedDict()
    categoricals = OrderedDict()
    source_fields = []

    for (k, v) in fields.iteritems():
        if v.__class__ is not dict:
            raise ValueError("Incorrect field specification: field "
                             "specifications are dictionaries that must "
                             "include a type definition, ex. "
                             "{'Phone': {type: 'String'}}"
                             )
        elif 'type' not in v:

            raise ValueError("Missing field type: field "
                             "specifications are dictionaries that must "
                             "include a type definition, ex. "
                             "{'Phone': {type: 'String'}}"
                             )
        elif v['type'] not in ['String',
                               'LatLong',
                               'Set',
                               'Source',
                               'Categorical',
                               'Custom',
                               'Interaction']:

            raise ValueError("Invalid field type: field "
                             "specifications are dictionaries that must "
                             "include a type definition, ex. "
                             "{'Phone': {type: 'String'}}"
                             )
        
        elif v['type'] != 'Custom' and 'comparator' in v :
            raise ValueError("Custom comparators can only be defined "
                             "for fields of type 'Custom'")

        elif v['type'] == 'Custom' and 'comparator' not in v :
            raise ValueError("For 'Custom' field types you must define "
                             "a 'comparator' function in the field "
                             "definition. ")


        elif v['type'] == 'LatLong' :
            v['comparator'] = compareLatLong

        elif v['type'] == 'Set' :
            v['comparator'] = compareJaccard

        elif v['type'] == 'String' :
            v['comparator'] = normalizedAffineGapDistance


        elif v['type'] == 'Categorical' :
            if 'Categories' not in v :
                raise ValueError('No "Categories" defined')

            comparator = CategoricalComparator(v['Categories'])

            for value, combo in sorted(comparator.combinations[2:]) :
                categoricals[str(combo)] = {'weight' : 0,
                                            'type' : 'Higher Categories',
                                            'value' : value}
            v['comparator'] = comparator
        

        elif v['type'] == 'Source' :
            if 'Source Names' not in v :
                raise ValueError('No "Source Names" defined')
            if len(v['Source Names']) != 2 :
                raise ValueError("You must supply two and only two source names")  
            source_fields.append(k)

            comparator = CategoricalComparator(v['Source Names'])

            for value, combo in sorted(comparator.combinations[2:]) :
                categoricals[str(combo)] = {'weight' : 0,
                                                'type' : 'Higher Categories',
                                                'value' : value}
                source_fields.append(str(combo))

            v['comparator'] = comparator
            

        elif v['type'] == 'Interaction' :
            if 'Interaction Fields' not in v :
                raise ValueError('No "Interaction Fields" defined')
                 
            for field in v['Interaction Fields'] :
                if fields[field].get('Has Missing') :
                    v.update({'Has Missing' : True})
                    break

            v.update({'weight': 0})
            interaction_terms[k] = v
            # We want the interaction terms to be at the end of of the
            # ordered dict so we'll add them after we finish
            # processing all the other fields
            continue
            
        

        data_model['fields'][k] = v

    data_model['fields'] = OrderedDict(data_model['fields'].items() 
                                       + categoricals.items()
                                       + interaction_terms.items())


    for k, v in data_model['fields'].items() :
        if k not in source_fields :
            if data_model['fields'][k].get('Has Missing') :
                missing = True
            else :
                missing = False
                
            for source_field in source_fields :
                if data_model['fields'][source_field].get('Has Missing') :
                    missing = True
            
                if v['type'] == 'Interaction' :
                    interaction_fields = [source_field]
                    interaction_fields += v['Interaction Fields']
                else :
                    interaction_fields = [source_field, k]

                data_model['fields'][source_field + ':' + k] =\
                  {'type' : 'Interaction', 
                   'Interaction Fields' : interaction_fields,
                   'Has Missing' : missing}


    for k, v in data_model['fields'].items() :
        if v.get('Has Missing') :
            data_model['fields'][k + ': not_missing'] = {'weight' : 0,
                                                         'type'   : 'Missing Data'}
        else :
            data_model['fields'][k].update({'Has Missing' : False})

     

    data_model['bias'] = 0
    return data_model
