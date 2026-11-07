# Code Quality and Linting

- **Adhere to Code Quality Standards:** Ensure all modifications comply with both general and language-specific linting rules.
- For Markdown files, ensure:
  - Fenced code blocks have language specified
  - Emphasis uses asterisks (`*text*`) not underscores (`_text_`)
  - Headers follow proper hierarchy
  - Links are properly formatted
- For code files, follow language-specific linting rules
- Use `read_lints` tool or IDE linter integration to check for issues

## Python Conventions

- **Docstrings:** terse and short. Never record the history of changes or why they were made; that belongs in commit messages and `CHANGELOG.md`.
- **Comments:** short, terse comments throughout each function marking the purpose of sections and snippets.
- **Typing:** variables are strictly typed. For values that are optional or not given, use empty typed values (e.g. `np.empty((0, 0))`, `""`, `[]`, `{}`) instead of `None` with `assert`/`is None` checks.
