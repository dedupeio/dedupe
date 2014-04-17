try:
    from collections import OrderedDict
except ImportError :
    from backport import OrderedDict

from dedupe.distance.affinegap import normalizedAffineGapDistance
from dedupe.distance.haversine import compareLatLong
from dedupe.distance.jaccard import compareJaccard
from dedupe.distance.categorical import CategoricalComparator


class DataModel(dict) :
    def __init__(self, fields):

        self['bias'] = 0
        self.comparison_fields = []

        self['fields'], source_fields = self.assignComparators(fields)

        self.higherCategoricals(source_fields)

        self.missingData()
        
        self.fieldDistanceVariables()

        self.total_fields = (
            len(self.field_comparators)
            + sum((length - 2) for _, length in self.categorical_indices)
            + len(self.interactions) 
            + len(self.missing_field_indices))

    def assignComparators(self, fields) :
        field_model = OrderedDict()
        interaction_terms = OrderedDict()
        categoricals = OrderedDict()
        source_fields = []

        for field, definition in fields.iteritems():

            self.checkFieldDefinitions(definition)

            if definition['type'] == 'LatLong' :
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
                    categoricals[str(combo)] = {'type' : 'Higher Categories',
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
                    categoricals[str(combo)] = {'type' : 'Higher Categories',
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

                interaction_terms[field] = definition
                # We want the interaction terms to be at the end of of the
                # ordered dict so we'll add them after we finish
                # processing all the other fields
                continue
            
            field_model[field] = definition
            self.comparison_fields.append(field)

        field_model = OrderedDict(field_model.items()
                                  + categoricals.items()
                                  + interaction_terms.items())

        return field_model, source_fields


    def higherCategoricals(self, source_fields) :
        for field, definition in self['fields'].items() :
            if field not in source_fields :
                if self['fields'][field].get('Has Missing') :
                    missing = True
                else :
                    missing = False
                
                for source_field in source_fields :
                    if self['fields'][source_field].get('Has Missing') :
                        missing = True
            
                    if definition['type'] == 'Interaction' :
                        interaction_fields = [source_field]
                        interaction_fields += definition['Interaction Fields']
                    else :
                        interaction_fields = [source_field, field]

                    self['fields'][source_field + ':' + field] =\
                          {'type' : 'Interaction', 
                           'Interaction Fields' : interaction_fields,
                           'Has Missing' : missing}




    def missingData(self) :
        for field, definition in self['fields'].items() :
            if definition.get('Has Missing') :
                self['fields'][field + ': not_missing'] =\
                  {'type'   : 'Missing Data'}
            else :
                self['fields'][field].update({'Has Missing' : False})

        

    def checkFieldDefinitions(self, definition) :
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
                             "{'Phone': {type: 'String'}}")
        
        elif definition['type'] != 'Custom' and 'comparator' in definition :
            raise ValueError("Custom comparators can only be defined "
                             "for fields of type 'Custom'")
                
        elif definition['type'] == 'Custom' and 'comparator' not in definition :
                raise ValueError("For 'Custom' field types you must define "
                                 "a 'comparator' function in the field "
                                 "definition. ")


    def fieldDistanceVariables(self) :

        fields = self['fields']
        field_names = fields.keys()

        self.interactions = []
        self.categorical_indices = []

        self.field_comparators = [(field, fields[field]['comparator'])
                                  for field in self.comparison_fields]

    
        self.missing_field_indices = [i for i, (field, v) 
                                      in enumerate(fields.items())
                                      if v.get('Has Missing')]

        for field, definition in fields.items() :
            field_type = definition['type']
            if field_type == 'Interaction' :
                interaction_indices = []
                for interaction_field in definition['Interaction Fields'] :
                    interaction_indices.append(field_names.index(interaction_field))
                self.interactions.append(interaction_indices)
            if field_type in ('Source', 'Categorical') :
                self.categorical_indices.append((field_names.index(field), 
                                                 definition['comparator'].length))


