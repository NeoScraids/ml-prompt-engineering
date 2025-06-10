# ml-prompt-engineering

> Ejemplos y plantillas de prompt engineering y fine-tuning de modelos de lenguaje con PyTorch y Hugging Face.

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
Este repositorio sirve como **catálogo profesional** de técnicas de prompt engineering y **pipeline de fine-tuning** de modelos de lenguaje:
- **Diseño de prompts**: ejemplos, buenas prácticas y medición de efectividad.
- **Fine-tuning**: preparación de datasets, configuración de PyTorch/Hugging Face y scripts de entrenamiento.
- **Reutilizable**: librerías en `src/`, notebooks interactivos y pruebas automatizadas.

---

## Requisitos
- Python 3.8+ y Conda
- CUDA toolkit (opcional para GPU)
- PyTorch y Transformers de Hugging Face

Instala el entorno:
```bash
conda env create -f environment.yml
conda activate ml-prompt-engineering
pip install -r requirements.txt
```

---

## Estructura de Archivos

- **.gitignore**: ignora entornos, caches y checkpoints.
- **environment.yml**: definición del entorno Conda.
- **requirements.txt**: dependencias pip.
- **data/**: datasets de ejemplos y para fine-tuning en formato JSONL.
- **notebooks/**: 
  - `prompt_design.ipynb`: explora ejemplos de prompts y evalúa resultados.
  - `fine_tuning.ipynb`: configura y lanza fine-tuning interactivo.
- **src/**: código modular:
  - `prompts.py`: funciones para generar y evaluar prompts.
  - `model_utils.py`: carga de modelos, tokenización y métricas.
  - `train.py`: script CLI para entrenamiento batch.
- **tests/**: `test_prompts.py` con PyTest para validar calidad de prompts.

---

## Prompt Design
En `notebooks/prompt_design.ipynb` encontrarás:
1. Ejemplos de **zero-shot**, **few-shot** y **chain-of-thought**.
2. Medición de logits y probabilidades para comparar plantillas.
3. Técnicas de **prompt tuning** y **soft prompts**.

### Ejemplo básico en `src/prompts.py`
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_completion(prompt, model_name="gpt2", max_length=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Fine-Tuning
Usando `notebooks/fine_tuning.ipynb` y `src/train.py`:
```bash
python src/train.py \
  --model_name_or_path gpt2 \
  --train_file data/fine_tuning_dataset.jsonl \
  --output_dir outputs/fine-tuned-model \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4
```
Descripción de parámetros:
- `model_name_or_path`: nombre del modelo base.
- `train_file`: ruta al JSONL con pares `prompt`/`completion`.
- `output_dir`: carpeta donde guardar checkpoints.

---

## Testing
Pruebas básicas con PyTest:
```bash
pytest tests/test_prompts.py
```

---

## Contribuciones
1. Crea un fork y una rama (`feature/tu-mejora`).
2. Añade tu prompt o script en la estructura correspondiente.
3. Ejecuta pruebas y notebooks.
4. Abre un Pull Request describiendo tu aporte.

---

## Licencia
MIT — véase [LICENSE](LICENSE) para más detalles.
