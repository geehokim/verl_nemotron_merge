# Lessons Learned

- **Date**: 2026-02-23
  - **Failure mode**: Used an unquoted heredoc while writing markdown that contained backticks and `->`, which triggered shell command substitution/redirection and created stray files.
  - **Detection signal**: Shell output showed unexpected `command not found`/`Permission denied`, and untracked files (`en`, `extra_info.lang`, `extra_info.language`) appeared.
  - **Prevention rule**: Always use quoted heredoc delimiters (`<<'EOF'`) when writing markdown/code blocks that contain shell-special characters.
