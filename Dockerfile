FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=2 PORT=7860 ENABLE_EVALUATION=false
WORKDIR /app
RUN useradd --create-home --uid 1000 appuser
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt
COPY --chown=appuser:appuser app.py /app/app.py
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser configs/config.yaml configs/classes.json /app/configs/
COPY --chown=appuser:appuser data/disease_info.json /app/data/disease_info.json
COPY --chown=appuser:appuser data/locales /app/data/locales
COPY --chown=appuser:appuser assets/fonts /app/assets/fonts
COPY --chown=appuser:appuser checkpoints/best_model.pth /app/checkpoints/best_model.pth
RUN mkdir -p /app/uploads /app/outputs/heatmaps /app/outputs/reports && chown -R appuser:appuser /app/uploads /app/outputs
USER appuser
EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
