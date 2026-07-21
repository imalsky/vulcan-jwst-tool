# Draft upstream report: PICASO 4.0.1 findings from the vulcan-jwst-tool integration

Status: DRAFT for Isaac's review. Nothing here has been posted anywhere;
posting (e.g. as GitHub issues on natashabatalha/picaso) requires explicit
approval. Each item is self-contained so it can be filed separately. All
measurements 2026-07-20/21 against picaso 4.0.1 and the v4.0 reference data
release.

## 1. Corrupted row in the Visscher 2121 chemistry grid (data)

File: `chemistry/visscher_grid_2121/sonora_2121grid_feh1.0_co0.55.txt`,
row at T = 900.0 K, log10 P = -5.523 (file row 925, counting data rows from
0 after the two header lines).

Anatomy: every reported species in the row is uniformly deflated by a
factor ~0.747 relative to the interpolation of its T-neighbors (H2 0.7471,
He 0.7471, H2O 0.7477, CO 0.7474, N2 0.7471, Na 0.7467, K 0.7466 ...), so
the gas-phase sum is 0.746 where every neighboring row sums to >= 0.9987.
Two species additionally carry junk residues: VO is ~9.9e6x too high
(5.2e-12 vs ~5e-19 expected from neighbors) and CrH ~4.8e4x (9.2e-13).
The same (T, P) cell is clean in the four neighboring composition files
(feh0.7_co0.55, feh1.5_co0.55, feh1.0_co0.46, feh1.0_co0.82).

The pattern suggests a spurious ~25% phantom abundance entered this row's
normalization during generation (deflating every reported species), with
corrupted VO/CrH values as residues of the same event.

## 2. chemeq_visscher_2121 docstring says 20 pressures; the files carry 21

The docstring block ("2020 data points: 20 pressures ... 101 temperatures")
disagrees with the shipped files, which are 21 pressures x 101 temperatures
= 2121 rows (log10 P from -6.0 to +4.0). Cosmetic, but the stated grid
shape is load-bearing for anyone validating a re-implementation.

## 3. Feature request: denser C/O sampling near the low-pressure CH4/H2O transition

At low pressure the equilibrium CH4/H2O transition is sharp and sits inside
the [0.55, 0.82] C/O cell: at 1 mbar / 1100 K the per-cell table slopes
d log10 X / d ln(C/O) are CH4 +1.24 (cell 0.46-0.55) vs +9.56 (cell
0.55-0.82), H2O -0.94 vs -9.31, CO2 -0.64 vs -9.07. Any interpolation
across the existing nodes therefore cannot produce a trustworthy local
composition derivative near C/O ~ 0.55-0.82 at low pressure (we evaluated
monotone-cubic interpolation as an alternative to linear; its node
derivatives are interpolant convention rather than data, and its
leave-one-node-out error is worse near the transition). One or two extra
nodes in (0.55, 0.82) would resolve this for derivative-based applications.

## 4. Native transmission silently returns all-NaN when gravity() gets bare gravity (code)

`case.gravity(gravity=..., gravity_unit=..., radius=...)` followed by
`case.spectrum(opa, calculation='transmission')` returns transit_depth =
all NaN: `atmsetup.get_altitude` computes g = G * planet.mass / z^2, and
`planet.mass` is NaN when only gravity+radius were provided
(constant_gravity is only forced when the RADIUS is NaN). Passing
mass + radius works. Suggestion: raise loudly in the transmission branch
when planet.mass is NaN instead of propagating NaN depths.

## 5. Observation (no action needed): find_strat keeps the guessed radiative-convective boundary

On a strongly irradiated planet (WASP-39b-like inputs, Tint 200 K,
rfacv 0.5), rcb guesses of 60/65/70/75 (91-level grid, 1e-6..300 bar) all
converge with the final convective zone starting exactly at the guess, all
Schwarzschild-consistent against the shipped adiabat table, with deep
temperatures differing by up to ~1000 K at 7.6 bar across the family
(shallower guesses fail flux balance and are correctly reported
unconverged). This appears to be the physical deep-adiabat degeneracy of
static irradiated RCE rather than a bug; we note it because "converged"
output can differ substantially at depth depending on the rcb guess, which
users of the climate mode may not expect.
