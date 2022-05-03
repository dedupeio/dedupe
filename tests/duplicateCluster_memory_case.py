import random
import dedupe.core
import dedupe.dedupe  # noqa: F401

# simulated_candidates = (((1, {'name': 'asdffdsa'}), (2, {'name': 'fdsaasdf'}))
# for _ in xrange(10**6))

# data_model =  {"fields": {"name": {"type": "String", "weight": -1.0}},
# "bias": 1.0}
# threshold = 0

# dupes = dedupe.core.scoreDuplicates(simulated_candidates,
# data_model,
# 0)

# simulated_candidates = (((1, {'name': 'asdffdsa'}), (2, {'name': 'fdsaasdf'}))
# for _ in xrange(10**7))


# deduper = dedupe.dedupe.Dedupe({"name": {"type": "String", "weight": -1.0}})
# clusters = deduper.duplicateClusters(simulated_candidates, 0, 0)


def candidates_gen():
    candidate_set = set([])
    for _ in range(10**5):
        block = [((random.randint(0, 1000), "a"), (random.randint(0, 1000), "b"))]
        for candidate in block:
            pair_ids = (candidate[0][0], candidate[1][0])
            if pair_ids not in candidate_set:
                yield candidate
                candidate_set.add(pair_ids)
    del candidate_set


@profile  # noqa: F821
def generator_test():
    a = sum(candidate[0][0] for candidate in candidates_gen())
    print(a)


generator_test()
