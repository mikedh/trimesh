# Bleep blorp this is the `trimesh` AI policy

Contributions are welcome, and AI is obviously a great tool! However there are a few notes:

- Please only open one PR at a time.
- Please keep AI text in written issues and PR bodies to 50 words or less.
- Clearly AI generated PRs and issues will probably get AI generated responses.
- API changes, dependency additions, "full algorithm rewrites" without prior discussion are pretty unlikely to be merged.
- Please read the `contributing.md` guide for instructions on surgical fixes with maximal test coverage.


Before comitting, run a self-review:
> Can you do a DEEP dive into the changes in this branch compared to `main`, and look for suspicious patterns: not using `numpy` operations, loops of any sort, nested loops, conditionals in loops, calling functions (or god help us, functions in loops). But also check the size of the work and profile heavily with pyinstrument or similar! Clean vectorized numpy can still be very slow, for example cross products against every face are going to be slower than pretty much everything else. Run the prior tests code under `pyinstrument` and compare it to your changes.
