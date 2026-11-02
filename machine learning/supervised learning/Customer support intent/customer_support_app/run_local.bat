@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements.txt
echo Launching Streamlit dashboard locally on your desktop...
streamlit run app.py
pause
