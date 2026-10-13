# ml-prompt-engineering

Dos scripts chicos para jugar con modelos de lenguaje: un wrapper para generar texto con cualquier modelo causal de HuggingFace (gpt2 por defecto) y un script que grafica loss y perplexity de un entrenamiento.

## Como probarlo

```bash
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Necesita Python 3.10 o superior.

Generar texto:

```python
from src.prompts import generate_completion

print(generate_completion(
    "Explain the difference between Kubernetes and Docker in simple terms:",
    max_new_tokens=80,
    temperature=0.7,
))
```

Usa GPU si hay (`cuda` o `mps` en Mac), si no cae a CPU. La primera vez baja el modelo de HuggingFace. `max_new_tokens` cuenta solo los tokens generados, no el prompt.

Graficar metricas con datos de ejemplo:

```bash
python plot_metrics.py
```

Guarda `metrics_plot.png` con dos paneles: loss por epoca y perplexity (`exp(loss)`). Para usar tus propios datos importa `plot_metrics` y pasale un dict con `loss` y `val_loss`.

## Estructura

```
src/prompts.py     generacion de texto
plot_metrics.py    graficas de loss y perplexity
requirements.txt
```

## Licencia

MIT
