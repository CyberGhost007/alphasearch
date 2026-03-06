# Contributing to AlphaSearch

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/CyberGhost007/alphasearch.git
cd alphasearch
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test your changes locally with `python server.py`
5. Commit with a clear message: `git commit -m "feat: add your feature"`
6. Push to your fork: `git push origin feature/your-feature`
7. Open a Pull Request

## Commit Messages

Use conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding or updating tests

## Code Style

- Follow existing patterns in the codebase
- Use type hints for function signatures
- Add docstrings to public functions and classes
- Keep functions focused and small

## Reporting Issues

- Use GitHub Issues
- Include steps to reproduce
- Include Python version and OS
- Include relevant error messages/logs (redact any API keys)

## Questions?

Open a Discussion or Issue on GitHub.
