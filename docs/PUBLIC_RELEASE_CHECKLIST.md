# Public Release Checklist

Use this checklist before pushing updates to the public repository.

- [x] Raw data excluded
- [x] `.env` files excluded
- [x] Local machine paths removed
- [x] Access tokens removed
- [x] Private manuscript and submission files excluded
- [x] Debug scripts and internal iteration files excluded
- [x] Selected figures reviewed for public release
- [x] Sample tables limited to small real-data extracts, not full result tables
- [x] Python scripts parse successfully
- [x] Documentation explains data access limits

Recommended Git workflow:

```bash
git status
git add README.md docs/ examples/ scripts/ figures/selected/
git commit -m "Describe the specific update"
git push
```
