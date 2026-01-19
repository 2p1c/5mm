---
name: commit
description: Commit changes to the git repository following best practices
---

# Commit Skill

Use this skill when the user asks to "commit" changes or runs `/commit`.

## Process

1.  **Check Status**: Run `git status` to see what files are changed.
2.  **Review Changes**: Run `git diff` (and `git diff --cached` if needed) to understand the changes.
3.  **Confirm Scope**: If there are many unrelated changes, ask the user if they want to split them or commit all.
4.  **Stage**: Use `git add` to stage the files intended for the commit.
5.  **Commit**: Generate a commit message following **Conventional Commits** format.

## Commit Message Format

Use the following structure:

```text
<type>(<scope>): <subject>

<body>
```

Common types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semi-colons, etc.
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintenance tasks

## Implementation Details

When executing the commit command, ALWAYS use the HEREDOC format to ensure safe string handling:

```bash
git commit -m "$(cat <<'EOF'
feat: add new functionality

Detailed description of the changes.
EOF
)"
```
