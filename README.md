# Proyecto 17: Trading algoritmico con Reinforcement Learning

Sistema de trading con RL (PPO) comparado con modelos predictivos clasicos (GARCH, PyMC, LSTM/GRU) y buy-and-hold.

## Activos
SPY, QQQ, TLT, GLD.

## Estructura
- data/: datos crudos y procesados.
- notebooks/: EDA, features, modelos, RL, evaluacion e interpretabilidad.
- src/: modulos reutilizables.
- reports/: informe tecnico y figuras.
- dashboard/: dashboard Plotly.
- models/: modelos entrenados.
- tests/: pruebas.

## Ejecucion
pip install -r requirements.txt
python create_project_zip.py
