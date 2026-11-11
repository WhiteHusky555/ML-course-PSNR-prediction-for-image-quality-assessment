# Лог прогона 02. Разведочный анализ

Пояснения к прогону — в [README.md](README.md). Время в начале строки отсчитывается от запуска ядра.

```text
     54.0s  {"python": "3.13.18", "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3", "scikit-image": "0.25.2", "opencv-python": "4.14.0.94", "opencv-python-headless": "4.14.0.94", "pyyaml": "6.0.3", "pytest": "8.4.2", "matplotlib": "3.10.0", "pillow": "13.0.0", "nbconvert": "6.4.5", "nbformat": "5.11.1", "ipykernel": "6.17.1", "jupyter_client": "7.4.9", "notebook": "6.5.7", "jupyterlab": "4.5.11"}
     56.6s  /usr/local/lib/python3.13/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
     56.6s    cells[i][c] = re.sub('\\\\\|', '|', cell)
     57.0s  /usr/local/lib/python3.13/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
     57.0s    text = re.sub(r'_', '\_', text) # Escape underscores in display text
     59.0s  [NbConvertApp] Converting notebook eda.ipynb to notebook
     60.2s  0.00s - Debugger warning: It seems that frozen modules are being used, which may
     60.2s  0.00s - make the debugger miss breakpoints. Please pass -Xfrozen_modules=off
     60.2s  0.00s - to python to disable frozen modules.
     60.2s  0.00s - Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1 to disable this validation.
     60.9s  0.00s - Debugger warning: It seems that frozen modules are being used, which may
     60.9s  0.00s - make the debugger miss breakpoints. Please pass -Xfrozen_modules=off
     60.9s  0.00s - to python to disable frozen modules.
     60.9s  0.00s - Note: Debugging will proceed. Set PYDEVD_DISABLE_FILE_VALIDATION=1 to disable this validation.
    318.5s  [NbConvertApp] Writing 1511195 bytes to /kaggle/working/eda.ipynb
    322.4s  [NbConvertApp] Converting notebook __script__.ipynb to html
    323.0s  [NbConvertApp] Writing 400959 bytes to __results__.html
```
