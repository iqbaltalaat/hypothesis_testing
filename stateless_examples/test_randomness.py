import itertools
import math
import statistics
import unittest
from randomness import generate

algos = ("mersenne", "randu", "middlesquare", "naive")
#algo = algos[-4]
algo = algos[0]

class TestMersenne(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, "algo"):
            self.algo = "mersenne"
        self.samples = [generate(algo=self.algo) for x in range(100000)]

    def test_repetition(self):
        seen = set()
        for i in range(1000):
            selection = self.samples[i:i+1000]
            new_hash = hash(tuple(selection))
            assert new_hash not in seen
            seen.add(new_hash)

    def test_median(self):
        assert 45 <= statistics.median(self.samples) <= 55

    def test_1d_uniformity(self):
        for ref, actual in zip(range(10, 100, 10), statistics.quantiles(self.samples, n=10)):
            assert (ref - 2.5) <= actual <= (ref +2.5)

    def test_nd_uniformity(self):
        # Wilson-Hilferty cube-root approximation; accurate for df >= 30
        def _chi2_pvalue(stat, df):
            z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
            return 0.5 * math.erfc(z / math.sqrt(2))

        val_min = min(self.samples)
        val_span = max(self.samples) - val_min + 1

        for n_dims in (2, 3):
            n_bins = 10 if n_dims == 2 else 5
            n_points = len(self.samples) // n_dims

            # Build n-dimensional histogram from consecutive n-tuples
            counts = {}
            for i in range(n_points):
                point = self.samples[i * n_dims : (i + 1) * n_dims]
                cell = tuple(
                    min(int((v - val_min) / val_span * n_bins), n_bins - 1)
                    for v in point
                )
                counts[cell] = counts.get(cell, 0) + 1

            n_cells = n_bins ** n_dims
            expected = n_points / n_cells
            chi2 = sum(
                (counts.get(cell, 0) - expected) ** 2 / expected
                for cell in itertools.product(range(n_bins), repeat=n_dims)
            )

            df = n_cells - 1
            p = _chi2_pvalue(chi2, df)
            assert p >= 0.01, (
                f"{n_dims}D uniformity failed for {self.algo}: "
                f"chi2={chi2:.1f} df={df} p={p:.4f}"
            )

class TestRandu(TestMersenne):
    algo = "randu"

class TestMiddleSquare(TestMersenne):
    algo = "middlesquare"

class TestNaive(TestMersenne):
    algo = "naive"

if __name__ == "__main__":
    unittest.main()
