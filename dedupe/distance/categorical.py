import itertools
import numpy

class CategoricalComparator(object):
    def __init__(self, category_names) :
        categories = [(name, name) for name in category_names]
        categories += list(itertools.combinations(category_names, 2))

        self.length = len(categories)

        self.categories = dict(zip(categories, itertools.count()))
        for k, v in self.categories.items() :
            self.categories[tuple(sorted(k, reverse=True))] = v

        values = [self.categories[combo] for combo in categories]

        self.combinations = zip(values,categories)

        self.categories_and_null = set(category_names + [''])

    def __call__(self, field_1, field_2):
        categories = (field_1, field_2)
        if categories in self.categories :
            return self.categories[categories]
        elif set(categories) <= self.categories_and_null :
            return numpy.nan
        else :
            raise ValueError("field not in Source Names")
