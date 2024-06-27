import io
import os
import time
from itertools import combinations

import dedupe
from benchmarks import common


def make_report(data, clustering):
    true_dupes = common.get_true_dupes(data)
    predicted_dupes = set([])
    for cluser_id, _ in clustering:
        for pair in combinations(cluser_id, 2):
            predicted_dupes.add(frozenset(pair))

    return common.Report.from_scores(true_dupes, predicted_dupes)


class Canonical:
    settings_file = common.DATASETS_DIR / "canonical_learned_settings"
    data_file = common.DATASETS_DIR / "restaurant-nophone-training.csv"

    def setup(self):
        self.data = common.load_data(self.data_file)
        training_pairs = dedupe.training_data_dedupe(self.data, "unique_id", 5000)
        self.training_pairs_filelike = io.StringIO()
        dedupe.serializer.write_training(training_pairs, self.training_pairs_filelike)
        self.training_pairs_filelike.seek(0)

    def make_report(self, clustering):
        return make_report(self.data, clustering)

    def run(self, use_settings=False):
        deduper: dedupe.StaticDedupe | dedupe.Dedupe

        if use_settings and os.path.exists(self.settings_file):
            with open(self.settings_file, "rb") as f:
                deduper = dedupe.StaticDedupe(f)

        else:
            variables = [
                dedupe.variables.String("name"),
                dedupe.variables.Exact("name"),
                dedupe.variables.String("address"),
                dedupe.variables.ShortString("cuisine", has_missing=True),
                dedupe.variables.ShortString("city"),
            ]

            deduper = dedupe.Dedupe(variables, num_cores=5)
            deduper.prepare_training(
                self.data, training_file=self.training_pairs_filelike, sample_size=10000
            )
            deduper.train(index_predicates=True)
            with open(self.settings_file, "wb") as f:
                deduper.write_settings(f)

        return deduper.partition(self.data, threshold=0.5)

    def time_run(self):
        return self.run()

    def peakmem_run(self):
        return self.run()

    def track_precision(self):
        return self.make_report(self.run()).precision

    def track_recall(self):
        return self.make_report(self.run()).recall


def cli():
    common.configure_logging()

    can = Canonical()
    can.setup()

    t0 = time.time()
    clustering = can.run(use_settings=True)
    elapsed = time.time() - t0

    print(can.make_report(clustering))
    print(f"ran in {elapsed} seconds")


if __name__ == "__main__":
    cli()
