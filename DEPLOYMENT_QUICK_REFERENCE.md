# Quick Reference: Deploy to PythonAnywhere

## 🚀 Initial Deployment (One-Time Setup)

### 1. Clone & Setup
```bash
cd ~
git clone https://github.com/Ayodeji-Samuel/scAdvisor-v2.git
cd scAdvisor-v2
mkvirtualenv --python=/usr/bin/python3.10 scadvisor-env
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
nano .env
```
Update:
- `SECRET_KEY` - Generate a new one
- `ALLOWED_HOSTS` - Add your PythonAnywhere domain
- `DEBUG=False`

### 3. Setup Django
```bash
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py createsuperuser
```

### 4. Configure Web App
- **Web Tab** → Add new web app → Manual configuration → Python 3.10
- **Virtualenv**: `/home/yourusername/virtualenvs/scadvisor-env`
- **WSGI file**: Edit with content from PYTHONANYWHERE_DEPLOYMENT.md
- **Static files**: 
  - URL: `/static/` → Directory: `/home/yourusername/scAdvisor-v2/staticfiles`
  - URL: `/media/` → Directory: `/home/yourusername/scAdvisor-v2/media`
- Click **Reload**

---

## 🔄 Update After GitHub Changes

```bash
cd ~/scAdvisor-v2
source ~/virtualenvs/scadvisor-env/bin/activate
git pull origin master
pip install -r requirements.txt           # if dependencies changed
python manage.py migrate                  # if models changed
python manage.py collectstatic --noinput  # if static files changed
```

Then: **Web Tab** → **Reload** button

---

## 🛠️ Useful Commands

### Check for issues
```bash
python manage.py check --deploy
```

### View error logs
Web Tab → Log files → error.log

### Restart web app
Web Tab → Reload button (green)

### Access Django shell
```bash
python manage.py shell
```

### Create new admin user
```bash
python manage.py createsuperuser
```

---

## ⚠️ Important Notes

1. **Always activate virtual environment** before running commands:
   ```bash
   source ~/virtualenvs/scadvisor-env/bin/activate
   ```

2. **Environment Variables**: Never commit `.env` file to GitHub

3. **Google Earth Engine**: Upload `ee-my-makinde-2b6858cddb01.json` manually to PythonAnywhere

4. **Database**: SQLite works for small projects. Upgrade to MySQL for production.

5. **Debug Mode**: Keep `DEBUG=False` in production

---

## 🔗 Links

- **GitHub Repo**: https://github.com/Ayodeji-Samuel/scAdvisor-v2
- **Live Site**: https://yourusername.pythonanywhere.com
- **Admin Panel**: https://yourusername.pythonanywhere.com/admin/

---

## 📞 Troubleshooting

**Site not loading?**
- Check Web tab for reload status
- View error logs
- Verify ALLOWED_HOSTS in .env

**Static files not showing?**
- Run `collectstatic`
- Check static file mappings in Web tab

**Database errors?**
- Run migrations
- Check db.sqlite3 file permissions

**Import errors?**
- Verify virtual environment path
- Reinstall requirements
