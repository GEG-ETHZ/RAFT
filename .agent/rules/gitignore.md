# Gitignore Formatting

**CRITICAL:** Git does **NOT** support inline comments in `.gitignore` files.

## Rules

- **Never use inline comments** - Comments must be on their own line
- **Place comments above patterns** - Use a separate line starting with `#`
- **Verify patterns work** - Use `git check-ignore -v <path>` to test

## Gitignore Examples

❌ **WRONG** - Inline comments don't work:

```gitignore
dist/        # Technical build output
node_modules/  # Dependencies
```

✅ **CORRECT** - Comments on separate lines:

```gitignore
# Technical build output
dist/
# Dependencies
node_modules/
```

## Why This Matters

Git treats the entire line (including spaces and comments) as the pattern. The pattern
`dist/        # comment` will NOT match a folder named `dist/` - it will try to match
a folder literally named `dist/        # comment`.

**Always test** your `.gitignore` patterns:

```bash
git check-ignore -v dist/
# Should output: .gitignore:15:dist/     dist/
```
