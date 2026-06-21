# Serve the Streamlit app on Hugging Face Spaces (Docker SDK). Python pinned to 3.12 so the
# recent numpy/pandas/scikit-learn wheels in requirements.txt resolve cleanly.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces routes to app_port (7860). Streamlit serves there, headless, bound to all interfaces.
EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]
