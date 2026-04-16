# Capability Scopes & Permissions

## 🌐 Browser Control

- **Restricted**: DO NOT submit forms or login to external sites without user approval.
- **Allowed**: You may use the headless browser to test the web app GUI, verify
  documentation links or fetch real-time library versions.

## 💻 Terminal Execution

### Host Machine

- **Restricted**: NEVER run `rm -rf`, use `sudo`, or modify system-level
  configuration (e.g., `/etc/`) without explicit approval.

### Guest VM (GVM)

- **Allowed**: Full administrative access is permitted for testing and
  provisioning. Destructive commands are allowed as the environment is disposable
  and designed for "fail fast" testing.

## General
### macOS (Host Machine)

- **Priority**: Always prioritize **Homebrew** (`brew install`) for installing
  system utilities (e.g., PDF creation, image manipulation).
- **Fallback**: Only suggest alternative methods (website, other repos) if the
  utility is not available via Homebrew.
