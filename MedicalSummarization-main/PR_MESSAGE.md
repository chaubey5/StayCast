Summary of changes:

- Frontend consolidation and theme refresh.
- Accessibility fixes and improved form semantics.
- API integration fixes for AJAX (CSRF and credentials) and JWT helper.
- Automated smoke tests used during development (removed after run).

Suggested reviewer notes:
- Verify register/login flows in a browser (dev server) and confirm JWT/token flows if switching to token-based auth.
- Review `frontend/static/css/app.css` for final brand colors and typography.
- Run `python manage.py check` and spot-check key pages.
