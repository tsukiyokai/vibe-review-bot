# AI Code Review

Automated code review bot for [CANN](https://gitcode.com/cann) repositories on GitCode. Powered by Claude Code's codereview skill.

## What It Does

- Fetches open pull requests from GitCode via API
- Runs AI-powered code review on PR diffs or local files
- Posts review comments back to GitCode PRs
- Tracks review adoption rates in a local SQLite database
- Polls for new pushes and reviews incrementally

## Usage

### Review PRs

```bash
# Set GitCode token
export GITCODE_TOKEN=your_token

# Review latest 3 open PRs (default repo: hcomm-dev)
python3 ai_reviewer.py

# Review specific PRs
python3 ai_reviewer.py --pr 1150 1144

# Review and post comments to GitCode
python3 ai_reviewer.py --pr 1150 --comment

# Review a different repo
python3 ai_reviewer.py --repo hcomm --pr 100

# Filter by author
python3 ai_reviewer.py --author lilin_137 -n 0
```

### Review Local Files

```bash
python3 ai_reviewer.py --file src/foo.cpp src/bar.h --save
```

### Continuous Polling

```bash
# Poll every 60s, review team members' PRs on new pushes
bash review_loop.sh $GITCODE_TOKEN 60 hcomm
```

The loop script tracks HEAD SHAs and only triggers reviews when changes are detected, keeping API usage minimal.

### Stats & Tracking

```bash
# View adoption rate stats
python3 ai_reviewer.py --stats

# Track review outcomes
python3 ai_reviewer.py --track --pr 1150

# Import historical review logs
python3 ai_reviewer.py --import-logs
```

## Project Structure

```
ai_reviewer.py    # Core reviewer: GitCode API, diff fetch, Claude invocation
review_loop.sh    # Polling daemon for continuous review
team.txt          # Team member list (name, ID, GitCode account)
skill/codereview  # Symlink to Claude Code codereview skill
log/              # Review output organized by repo/PR/file
```

## Configuration

- `team.txt` — whitespace-separated file with team members' GitCode accounts (used by the polling loop to filter PRs)
- `--repo` — targets both the local clone path (`~/repo/<repo>/`) and GitCode API (`cann/<repo>`)
