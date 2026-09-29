# Bleep blorp these are the `trimesh` AI preferences

Contributions are welcome, and AI is obviously a great tool! However there are a few notes:

- We're trying to keep every line human-read and human-run at least once, so the larger the diff the less likely a PR is to be merged.
- Please only open one PR at a time.
- Please keep AI text in written issues and PR bodies to 50 words or less.
- API changes, dependency additions, "full algorithm rewrites" without prior discussion are pretty unlikely to be merged.
- Please read the `contributing.md` guide for instructions on surgical fixes with maximal test coverage.


Before comitting, run a self-review:
> Can you do a DEEP dive, with deep research and subagents, into the changes in this branch compared to `main`, and look for suspicious patterns: not using `numpy` operations, loops of any sort, nested loops, conditionals in loops, calling functions, O(N^2) or worse anywhere. Also check the size of `N` and profile heavily with pyinstrument or similar! Clean vectorized numpy can still be very slow, for example cross products against every face are going to be slower than pretty much everything else. Run an example of the code under `pyinstrument` and compare it to your changes. Also check if there ANY opportunities to reduce the size of the diff? Merge likelyhood goes up dramatically for smaller changes. Make a careful, maximalist cleanup plan and propose it for execution.
