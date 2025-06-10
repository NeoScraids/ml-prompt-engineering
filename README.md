# ml-prompt-engineering

> Catálogo profesional de técnicas de prompt engineering y pipeline de fine-tuning con PyTorch y Hugging Face, enriquecido con métricas y visualizaciones.

```
ml-prompt-engineering/
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── requirements.txt
├── data/
│   ├── examples.csv
│   └── fine_tuning_dataset.jsonl
├── notebooks/
│   ├── prompt_design.ipynb
│   └── fine_tuning.ipynb
├── src/
│   ├── prompts.py
│   ├── model_utils.py
│   └── train.py
└── tests/
    └── test_prompts.py
```

---

## Descripción
Este repositorio ofrece:
- **Prompt Design**: ejemplos y pruebas de zero-shot, few-shot y chain-of-thought.
- **Fine-Tuning**: preparación de datasets JSONL y scripts de entrenamiento.
- **Métricas & Visualizaciones**: gráficos de loss, perplexity y comparación de probabilidades.
- **Modularidad**: librerías reusable en `src/`, notebooks interactivos y pruebas automatizadas.

---

## Requisitos
- Python 3.8+ con Conda
- CUDA toolkit (para GPU, opcional)
- PyTorch >=1.12 y Transformers

Instalación:
```bash
conda env create -f environment.yml
conda activate ml-prompt-engineering
pip install -r requirements.txt
```

---

## Estructura de Archivos
- `.gitignore`: ignora caches, entornos y checkpoints.
- `environment.yml` y `requirements.txt`: configuración de entorno.
- `data/`: ejemplos (`examples.csv`) y dataset de fine-tuning (`.jsonl`).
- `notebooks/`: análisis interactivo:
  - **prompt_design.ipynb**: diseña y evalúa prompts.
  - **fine_tuning.ipynb**: entrena modelos y visualiza métricas.
- `src/`: código:
  - `prompts.py` → generación y evaluación de prompts.
  - `model_utils.py` → carga/modelo/tokenización/métricas.
  - `train.py` → CLI batch training con registro de métricas.
- `tests/`: validaciones con PyTest.

---

## Prompt Design
En `notebooks/prompt_design.ipynb`:
1. **Zero-shot** y **few-shot** con ejemplos.
2. **Chain-of-Thought** para mejorar razonamiento.
3. Medición de **probabilidades** y **logits**.

### Snippet de generación
```python
# src/prompts.py
def generate_completion(prompt, model_name="gpt2", max_length=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Fine-Tuning
Usa `src/train.py` o `notebooks/fine_tuning.ipynb`:
```bash
python src/train.py \
  --model_name_or_path gpt2 \
  --train_file data/fine_tuning_dataset.jsonl \
  --output_dir outputs/fine-tuned-model \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4
```

## Métricas y Visualizaciones
Se registran en entrenamiento:
- **Loss** por epoch/step.
- **Perplexity** variante de cross-entropy.

Ejemplo de código para plot en notebook:
```python
from matplotlib import pyplot as plt

def plot_metrics(history):
    epochs = range(1, len(history['loss'])+1)
    plt.figure()
    plt.plot(epochs, history['loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss por Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Perplexity = exp(loss)
    plt.figure()
    plt.plot(epochs, [math.exp(l) for l in history['loss']], label='Train Perplexity')
    plt.plot(epochs, [math.exp(l) for l in history['val_loss']], label='Val Perplexity')
    plt.title('Perplexity por Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.show()
```
Incluye ese bloque en tu notebook de fine-tuning para generar gráficos profesionales.

---

## Testing
```bash
pytest tests/test_prompts.py
```

---

## Contribuciones
1. Fork → rama `feature/tu-aporte`
2. Añade prompts/datasets/notebooks.
3. Corre pruebas y notebooks.
4. Pull Request detallado.

---

## Licencia
MIT — véase [LICENSE](LICENSE) para detalles.
