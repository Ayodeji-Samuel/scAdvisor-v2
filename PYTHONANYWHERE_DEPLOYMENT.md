# PythonAnywhere Deployment Guide for scAdvisor-v2

## Prerequisites
- PythonAnywhere account (free or paid)
- GitHub repository: https://github.com/Ayodeji-Samuel/scAdvisor-v2

## Step 1: Clone Repository

Open a **Bash console** on PythonAnywhere and run:

```bash
cd ~
git clone https://github.com/Ayodeji-Samuel/scAdvisor-v2.git
cd scAdvisor-v2
```

## Step 2: Create Virtual Environment

```bash
mkvirtualenv --python=/usr/bin/python3.10 scadvisor-env
```

If the above doesn't work, use:
```bash
python3.10 -m venv ~/virtualenvs/scadvisor-env
source ~/virtualenvs/scadvisor-env/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
nano ~/scAdvisor-v2/.env
```

Add the following (replace with your actual values):

```env
SECRET_KEY=your-new-secret-key-here-make-it-random-and-long
DEBUG=False
ALLOWED_HOSTS=yourusername.pythonanywhere.com
DATABASE_URL=sqlite:///db.sqlite3
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

## Step 5: Update Settings for Production

Your project should use the `settings_production.py` file. Make sure it's configured properly:

```bash
nano ~/scAdvisor-v2/smartproject/settings_production.py
```

## Step 6: Collect Static Files

```bash
cd ~/scAdvisor-v2
python manage.py collectstatic --noinput
```

## Step 7: Run Migrations

```bash
python manage.py migrate
```

## Step 8: Create Superuser

```bash
python manage.py createsuperuser
```

## Step 9: Configure Web App on PythonAnywhere

1. Go to the **Web** tab in PythonAnywhere
2. Click **Add a new web app**
3. Choose **Manual configuration** (NOT Django)
4. Select **Python 3.10**

### Configure the WSGI file:

Click on the WSGI configuration file link and replace its contents with:

```python
import os
import sys

# Add your project directory to the sys.path
path = '/home/yourusername/scAdvisor-v2'
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variable for Django settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'smartproject.settings_production'

# Activate virtual environment
from dotenv import load_dotenv
project_folder = os.path.expanduser('/home/yourusername/scAdvisor-v2')
load_dotenv(os.path.join(project_folder, '.env'))

# Import Django WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
```

**Replace `yourusername` with your actual PythonAnywhere username!**

## Step 10: Configure Virtual Environment

In the **Web** tab:
- Under **Virtualenv** section, enter: `/home/yourusername/virtualenvs/scadvisor-env`
- Replace `yourusername` with your actual username

## Step 11: Configure Static Files

In the **Web** tab, add these static file mappings:

| URL | Directory |
|-----|-----------|
| /static/ | /home/yourusername/scAdvisor-v2/staticfiles |
| /media/ | /home/yourusername/scAdvisor-v2/media |

## Step 12: Reload Web App

Click the green **Reload** button at the top of the Web tab.

## Step 13: Test Your Site

Visit: `https://yourusername.pythonanywhere.com`

---

## Updating Your Code (After Making Changes)

When you push updates to GitHub, run these commands on PythonAnywhere:

```bash
cd ~/scAdvisor-v2
git pull origin master
source ~/virtualenvs/scadvisor-env/bin/activate
pip install -r requirements.txt  # if requirements changed
python manage.py migrate  # if models changed
python manage.py collectstatic --noinput  # if static files changed
```

Then reload your web app from the Web tab.

---

## Troubleshooting

### Error Logs
Check error logs in the Web tab → Log files section

### Common Issues:

1. **Import errors**: Make sure virtual environment is activated
2. **Static files not loading**: Run `collectstatic` and check static file mappings
3. **Database errors**: Run migrations
4. **500 errors**: Check error logs and ensure DEBUG=False in production

### Test locally before deploying:
```bash
python manage.py check --deploy
```

---

## Security Checklist

- ✅ DEBUG = False in production
- ✅ SECRET_KEY is unique and not in version control
- ✅ ALLOWED_HOSTS is set correctly
- ✅ .env file is in .gitignore
- ✅ Use HTTPS (PythonAnywhere provides this automatically)
