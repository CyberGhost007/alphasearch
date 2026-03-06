# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainer or open a private security advisory on GitHub
3. Include steps to reproduce the vulnerability
4. Allow reasonable time for a fix before public disclosure

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes       |

## Security Considerations

- **API Keys**: Never commit `.env` files or API keys. Use `.env.example` as a template.
- **File Uploads**: PDF uploads are validated (magic bytes, size limits, extension checks).
- **Local Storage**: All data is stored locally in `.treerag_data/`. No external databases.
- **LLM Calls**: Queries are sent to OpenAI's API. Review OpenAI's privacy policy for data handling.
