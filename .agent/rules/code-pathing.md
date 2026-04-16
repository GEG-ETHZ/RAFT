# Code Pathing Rules

When writing, modifying, or suggesting code for this project, you must adhere to the following rules regarding file paths:

## 1. Default to Relative Paths

Your default behavior must **always** be to use relative paths (e.g., `../components/Button.tsx`, `./assets/logo.png`) in the codebase. Absolute paths (especially those tied to a specific developer's machine like `/Users/an/...`) break the code for other developers and CI/CD pipelines.

## 2. Exceptions for Absolute Paths

If there is an absolute necessity to use an absolute path in the codebase:

1. **Explicit Permission:** You must explicitly prompt the user and ask for permission, explaining *why* an absolute path is required.
2. **Code Comments:** If approved, you must add clear comments above the path in the code warning that it is an absolute path and explaining why it is necessary.
