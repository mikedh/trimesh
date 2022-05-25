---
name: Bug report
about: Default Issue
title: ''
labels: ''
assignees: ''

---

Thanks for reporting an issue! Please provide a minimal code block which reproduces the issue, ideally with an `assert` that should succeed but fails so we can add your issue to unit tests:

```
m = trimesh.creation.box()
assert m.faces.shape == (12, 3)
```

If the issue is model-specific please attach the file or a link to the file. Thanks!
