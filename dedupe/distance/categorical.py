import itertools

class SourceComparator(object):
    def __init__(self, source_names) :
        assert len(source_names) == 2

        sources = [(name, name) for name in source_names]
        sources += list(itertools.combinations(source_names, 2))
        self.sources = dict(zip(sources, itertools.count()))
        for k, v in self.sources.items() :
            self.sources[tuple(sorted(k, reverse=True))] = v

        self.sources_and_null = set(source_names + [''])
    def __call__(self, field_1, field_2):
        sources = (field_1, field_2)
        if sources in self.sources :
            return self.sources[sources]
        elif set(sources) <= self.sources_and_null :
            return numpy.nan
        else :
            raise ValueError("field not in Source Names")
