# Changelog

## Unreleased

- Consolidated frontend templates into top-level `frontend/templates/` and removed legacy `pages/` folder.
- Reworked header/footer into reusable includes.
- Applied refreshed dark SaaS theme with updated `frontend/static/css/app.css`.
- Fixed AJAX auth flows: added `credentials: 'same-origin'` and CSRF handling in `frontend/static/js/app.js`.
- Added JWT token helper `fetchToken` for optional token workflows.
- Improved accessibility (ARIA attributes) on navbar, footer, and form components.
- Added automated end-to-end smoke tests (run locally) for register/login/predict/contact.
