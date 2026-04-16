# Agent Pathing Rules

**CRITICAL:** The `.agent` directory is checked into version control and shared among all developers.

Therefore, when creating, editing, or referencing files within the `.agent` directory (such as rules or workflows), you **MUST NEVER** use absolute paths (e.g., `/Users/an/...`).

You must **ALWAYS** use relative paths (e.g., `./.agent/rules/...` or `./.agent/workflows/...`) to ensure the configuration remains portable and functional for every developer on the project regardless of their local machine setup.
