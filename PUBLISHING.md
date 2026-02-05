# Publishing to PyPI with Trusted Publishers

This guide walks you through setting up PyPI trusted publishing for this repo.

## Prerequisites

1. A **public** GitHub repository (required for OIDC trusted publishing)
2. PyPI account with the `driftrail` project approved

## Step 1: Create Public GitHub Repo

1. Go to https://github.com/new
2. Create a new **public** repository (e.g., `driftrail-python`)
3. Push this folder's contents to that repo:

```bash
cd driftrail-pypi
git init
git add .
git commit -m "Initial commit - DriftRail Python SDK v2.0.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/driftrail-python.git
git push -u origin main
```

## Step 2: Create GitHub Environments

In your GitHub repo, go to **Settings → Environments** and create two environments:

### Environment 1: `testpypi`
- No special protection rules needed

### Environment 2: `pypi`
- Recommended: Add "Required reviewers" for production safety
- This ensures someone approves before publishing to real PyPI

## Step 3: Configure PyPI Trusted Publishers

### For TestPyPI (optional but recommended):
1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - **PyPI Project Name**: `driftrail`
   - **Owner**: Your GitHub username or org
   - **Repository**: `driftrail-python` (or whatever you named it)
   - **Workflow name**: `publish.yml`
   - **Environment**: `testpypi`

### For PyPI:
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - **PyPI Project Name**: `driftrail`
   - **Owner**: Your GitHub username or org
   - **Repository**: `driftrail-python`
   - **Workflow name**: `publish.yml`
   - **Environment**: `pypi`

## Step 4: Publish a Release

### Option A: Tag-based release (recommended)
```bash
# Update version in pyproject.toml and driftrail/__init__.py first
git add .
git commit -m "Release v2.0.0"
git tag v2.0.0
git push origin main --tags
```

The workflow will automatically:
1. Build the package
2. Publish to TestPyPI
3. Publish to PyPI

### Option B: Manual trigger
1. Go to **Actions** tab in your GitHub repo
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"

## Updating the Package

When you want to release a new version:

1. Update version in `pyproject.toml` and `driftrail/__init__.py`
2. Update `CHANGELOG.md`
3. Commit and tag:
   ```bash
   git add .
   git commit -m "Release v2.1.0"
   git tag v2.1.0
   git push origin main --tags
   ```

## Troubleshooting

### "Publisher not found" error
- Ensure the workflow filename matches exactly (`publish.yml`)
- Ensure the environment name matches exactly (`pypi` or `testpypi`)
- Ensure the repository owner/name match exactly

### Build fails
- Run locally first: `pip install build && python -m build`
- Check that all files are committed

### Permission denied
- Verify the trusted publisher is configured correctly on PyPI
- Check that the GitHub environment exists and matches

## Security Notes

- The public repo only contains the SDK code, not your main DriftRail codebase
- No API keys or secrets are stored in this repo
- Trusted publishing uses short-lived OIDC tokens (no long-lived credentials)
- Consider adding branch protection rules to `main`
