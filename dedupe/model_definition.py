from dedupe.distance.affinegap import normalizedAffineGapDistance
from dedupe.distance.haversine import compareLatLong
from dedupe.distance.jaccard import compareJaccard
from dedupe.distance.categorical import CategoricalComparator

try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict


def initializeDataModel(fields):
    """Initialize a data_model with a field definition"""

    # The data model needs to order the different kinds of variabls in
    # a particular way. First, variables that are directly produced by
    # a comparison function have to go first. Second, dummy variable
    # that will be set from the return value of a comparison
    # function. Third, interaction variables of the first two
    # types. Finally, handle any missing data flags. The reasons for this
    # order has to do with implementation details of core.fieldDistances

    field_model = OrderedDict()

    interaction_terms = OrderedDict()
    categoricals = OrderedDict()
    source_fields = []

    for field, definition in fields.iteritems():
        if definition.__class__ is not dict:
            raise ValueError("Incorrect field specification: field "
                             "specifications are dictionaries that must "
                             "include a type definition, ex. "
                             "{'Phone': {type: 'String'}}"
                             )

        elif 'type' not in definition:
            raise ValueError("Missing field type: field "
                             "specifications are dictionaries that must "
                             "include a type definition, ex. "
                             "{'Phone': {type: 'String'}}"
                             )

        elif definition['type'] not in ['String',
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
        
        elif definition['type'] != 'Custom' and 'comparator' in definition :
            raise ValueError("Custom comparators can only be defined "
                             "for fields of type 'Custom'")

        elif definition['type'] == 'Custom' and 'comparator' not in definition :
            raise ValueError("For 'Custom' field types you must define "
                             "a 'comparator' function in the field "
                             "definition. ")

        elif definition['type'] == 'LatLong' :
            definition['comparator'] = compareLatLong

        elif definition['type'] == 'Set' :
            definition['comparator'] = compareJaccard

        elif definition['type'] == 'String' :
            definition['comparator'] = normalizedAffineGapDistance

        elif definition['type'] == 'Categorical' :
            if 'Categories' not in definition :
                raise ValueError('No "Categories" defined')

            comparator = CategoricalComparator(definition['Categories'])

            for value, combo in sorted(comparator.combinations[2:]) :
                categoricals[str(combo)] = {'weight' : 0,
                                            'type' : 'Higher Categories',
                                            'value' : value}

            definition['comparator'] = comparator
        
        elif definition['type'] == 'Source' :
            if 'Source Names' not in definition :
                raise ValueError('No "Source Names" defined')
            if len(definition['Source Names']) != 2 :
                raise ValueError("You must supply two and only " 
                                  "two source names")  
            source_fields.append(field)

            comparator = CategoricalComparator(definition['Source Names'])

            for value, combo in sorted(comparator.combinations[2:]) :
                categoricals[str(combo)] = {'weight' : 0,
                                            'type' : 'Higher Categories',
                                            'value' : value}
                source_fields.append(str(combo))

            definition['comparator'] = comparator
            
        elif definition['type'] == 'Interaction' :
            if 'Interaction Fields' not in definition :
                raise ValueError('No "Interaction Fields" defined')
                 
            for interacting_field in definition['Interaction Fields'] :
                if fields[interacting_field].get('Has Missing') :
                    definition.update({'Has Missing' : True})
                    break

            definition.update({'weight': 0})
            interaction_terms[field] = definition
            # We want the interaction terms to be at the end of of the
            # ordered dict so we'll add them after we finish
            # processing all the other fields
            continue
            
        field_model[field] = definition

    field_model = OrderedDict(field_model.items() 
                              + categoricals.items()
                              + interaction_terms.items())


    for field, definition in field_model.items() :
        if field not in source_fields :
            if field_model[field].get('Has Missing') :
                missing = True
            else :
                missing = False
                
            for source_field in source_fields :
                if field_model[source_field].get('Has Missing') :
                    missing = True
            
                if definition['type'] == 'Interaction' :
                    interaction_fields = [source_field]
                    interaction_fields += definition['Interaction Fields']
                else :
                    interaction_fields = [source_field, field]

                field_model[source_field + ':' + field] =\
                  {'type' : 'Interaction', 
                   'Interaction Fields' : interaction_fields,
                   'Has Missing' : missing}

    for field, definition in field_model.items() :
        if definition.get('Has Missing') :
            field_model[field + ': not_missing'] =\
              {'weight' : 0,
               'type'   : 'Missing Data'}
        else :
            field_model[field].update({'Has Missing' : False})

     
    
    data_model = {'fields' : field_model,
                  'bias' : 0}

    return data_model
