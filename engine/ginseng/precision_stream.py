"""Random-access rows of the existing precision PCG64 float64 stream.

Version 1 keeps seed domain 400 and the old fixed-material-width C-order layout.
One float64 uniform consumes one PCG64 64-bit output. Advancing by complete rows
preserves old streams; it does not invent a new sampler or PRNG.
"""

from dataclasses import dataclass

import numpy as np

from ginseng.sampling import derive_seed, validate_size


@dataclass(frozen=True)
class PrecisionStream:
    root_seed: int
    replicate: int
    material_horizon: int

    def __post_init__(self):
        validate_size("mc", 1, self.material_horizon)
        derive_seed(self.root_seed, "mc", self.replicate, 400)

    @property
    def dimension(self):
        return 2 * self.material_horizon - 1

    @property
    def seed(self):
        return derive_seed(self.root_seed, "mc", self.replicate, 400)

    def points(self, start, count):
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 1
            or start + count > 2**30
        ):
            raise ValueError("Invalid precision stream row range")
        validate_size("mc", count, self.material_horizon)
        bitgen = np.random.PCG64(self.seed)
        bitgen.advance(start * self.dimension)
        return np.random.Generator(bitgen).random((count, self.dimension))

    def identity(self):
        return dict(
            version=1,
            purpose="cash_failure_precision",
            domain=400,
            root_seed=self.root_seed,
            replicate=self.replicate,
            derived_seed=self.seed,
            bit_generator="PCG64",
            uniform_dtype="float64",
            material_horizon=self.material_horizon,
            dimension=self.dimension,
            layout="fixed-material C-order float64 rows v1",
            extension="Path-count/chunk/worker/visible-horizon prefixes are stable within this material layout. A different material horizon is a different layout.",
        )
