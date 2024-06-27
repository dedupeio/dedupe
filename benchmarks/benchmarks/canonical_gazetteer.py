import os
import time

import dedupe
from benchmarks import canonical_matching, common


def make_report(data, clustering):
    true_dupes = canonical_matching.get_true_dupes(data)
    predicted_dupes = set(
        frozenset([a, b]) for a, result in clustering for b, score in result
    )
    return common.Report.from_scores(true_dupes, predicted_dupes)


class Gazetteer(canonical_matching.Matching):
    settings_file = common.DATASETS_DIR / "canonical_gazetteer_learned_settings"
    data_1_file = common.DATASETS_DIR / "restaurant-1.csv"
    data_2_file = common.DATASETS_DIR / "restaurant-2.csv"

    params = [None]  # placholder

    def make_report(self, clustering):
        return make_report(self.data, clustering)

    def run(self, kwargs, use_settings=False):
        data_1, data_2 = self.data

        if use_settings and os.path.exists(self.settings_file):
            with open(self.settings_file, "rb") as f:
                gazetteer = dedupe.StaticGazetteer(f)
        else:
            variables = [
                {"field": "name", "type": "String"},
                {"field": "address", "type": "String"},
                {"field": "cuisine", "type": "String"},
                {"field": "city", "type": "String"},
            ]

            gazetteer = dedupe.Gazetteer(variables)
            gazetteer.prepare_training(
                data_1,
                data_2,
                training_file=self.training_pairs_filelike,
                sample_size=10000,
            )
            gazetteer.train()

            with open(self.settings_file, "wb") as f:
                gazetteer.write_settings(f)

        gazetteer.index(data_2)
        gazetteer.unindex(data_2)
        gazetteer.index(data_2)

        return gazetteer.search(data_1, n_matches=1, generator=True)


def cli():
    common.configure_logging()

    gaz = Gazetteer()
    gaz.setup(None)

    t0 = time.time()
    clustering = gaz.run(None, use_settings=True)
    elapsed = time.time() - t0

    print(gaz.make_report(clustering))
    print(f"ran in {elapsed} seconds")


if __name__ == "__main__":
    cli()
