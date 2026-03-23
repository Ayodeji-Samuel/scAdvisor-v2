# Production settings for PythonAnywhere
from .settings import *
import os

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Read SECRET_KEY from environment (set in .env or PythonAnywhere env vars)
SECRET_KEY = os.environ.get('SECRET_KEY', SECRET_KEY)

# Add your PythonAnywhere domain — override via ALLOWED_HOSTS env var
_allowed = os.environ.get('ALLOWED_HOSTS', '')
ALLOWED_HOSTS = [h.strip() for h in _allowed.split(',') if h.strip()] or ['*']

# ------------------------------------------------------------------
# CSRF fix: Django 4.0+ requires CSRF_TRUSTED_ORIGINS when DEBUG=False.
# Without this, every login/signup POST returns 403 Forbidden.
# PythonAnywhere serves over HTTPS, so the origin must use https://.
# ------------------------------------------------------------------
_csrf_origins = os.environ.get('CSRF_TRUSTED_ORIGINS', '')
CSRF_TRUSTED_ORIGINS = (
    [o.strip() for o in _csrf_origins.split(',') if o.strip()]
    or ['https://*.pythonanywhere.com']
)

# PythonAnywhere terminates SSL at its load balancer and forwards
# requests to Django over plain HTTP.  Tell Django to trust the
# X-Forwarded-Proto header so it knows the original request was HTTPS.
# This is also required for CSRF origin checking to work correctly.
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Database for production (you can use SQLite or upgrade to MySQL)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Static files settings
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')

# Media files settings
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Security settings for production
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Google Earth Engine settings
# Make sure your service account key is in the project root
GEE_SERVICE_ACCOUNT_KEY = os.path.join(BASE_DIR, 'ee-my-makinde-2b6858cddb01.json')

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'django.log'),
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Create logs directory
import pathlib
pathlib.Path(os.path.join(BASE_DIR, 'logs')).mkdir(exist_ok=True)
