
# Conceptos fundamentales de Word2Vec

## 1. El problema: cómo representar palabras numéricamente

Muchos algoritmos de Machine Learning requieren **vectores numéricos** como entrada.
Por lo tanto, necesitamos transformar palabras en números.

Una representación simple es **One-Hot Encoding**.

Supongamos el vocabulario:

["rey", "reina", "hombre", "mujer"]

Entonces:

rey    = [1,0,0,0]  
reina  = [0,1,0,0]  
hombre = [0,0,1,0]  
mujer  = [0,0,0,1]

### Problema

Esta representación tiene dos limitaciones importantes:

1. Alta dimensionalidad.
2. No captura similitud semántica.

Por ejemplo:

cos(rey, reina) = 0  
cos(rey, mesa) = 0

El modelo no distingue entre palabras similares y no relacionadas.

---

# 2. Hipótesis distribucional del lenguaje

Word2Vec se basa en una idea clásica de lingüística:

**Las palabras que aparecen en contextos similares tienden a tener significados similares.**

Ejemplo:

El rey gobierna el reino  
La reina gobierna el reino

Los contextos de **rey** y **reina** son similares.

Entonces el modelo aprende vectores tales que:

vector(rey) ≈ vector(reina)

---

# 3. Qué es un embedding

Un **embedding** es una representación vectorial densa de una palabra.

Ejemplo:

rey   → [0.34, -0.21, 0.88, ...]  
reina → [0.36, -0.19, 0.90, ...]

Características:

- dimensión baja (50–300 típicamente)
- valores continuos
- capturan relaciones semánticas

---

# 4. Word2Vec: la idea central

Word2Vec aprende embeddings entrenando una **red neuronal simple** que intenta predecir palabras usando su contexto.

Hay dos arquitecturas principales:

1. **CBOW (Continuous Bag of Words)**
2. **Skip-gram**

---

# 5. Ventana de contexto

Word2Vec utiliza una **ventana de contexto** alrededor de cada palabra.

Ejemplo:

El rey gobierna el reino

Con ventana = 2:

centro: gobierna  
contexto: [el, rey, el, reino]

---

# 6. Arquitectura CBOW

### Idea

Predecir la **palabra central** a partir de su contexto.

(el, rey, el, reino) → gobierna

### Arquitectura

context words → embeddings → average → softmax → predicted word

Formalmente:

x → W → h → W' → y

donde

W  = matriz de embeddings  
W' = matriz de salida

---

# 7. Arquitectura Skip-gram

### Idea

Predecir el **contexto** a partir de la **palabra central**.

Ejemplo:

gobierna → rey  
gobierna → reino

Training samples:

(centro, contexto)

Ejemplo:

(rey, gobierna)  
(rey, reino)

---

# 8. Arquitectura neuronal de Word2Vec

Supongamos:

V = tamaño del vocabulario  
D = dimensión del embedding

La red tiene:

### Entrada

One-hot vector de dimensión V

### Capa oculta

h = Wᵀ x

donde

W ∈ R^(V×D)

Esto selecciona el embedding de la palabra.

### Capa de salida

u = W' h

donde

W' ∈ R^(D×V)

Luego se aplica softmax para obtener probabilidades de palabras.

---

# 9. Qué aprende realmente la red

El objetivo del entrenamiento es maximizar:

P(contexto | palabra)

Durante el entrenamiento:

- W se convierte en la matriz de embeddings
- cada fila de W es el vector de una palabra

---

# 10. Problema computacional

El softmax requiere calcular V probabilidades.

Si el vocabulario tiene

V = 100,000

esto es costoso.

---

# 11. Negative Sampling

Para evitar calcular el softmax completo, Word2Vec usa **Negative Sampling**.

La idea es convertir el problema en una **clasificación binaria**.

Para cada par real:

(rey, reino)

generamos pares falsos:

(rey, mesa)  
(rey, coche)  
(rey, montaña)

El modelo aprende a distinguir:

contexto real vs contexto falso.

---

# 12. Distribución de Negative Sampling

Las palabras negativas se muestrean con:

P(w) ∝ f(w)^0.75

donde f(w) es la frecuencia de la palabra.

Esto reduce el impacto de palabras extremadamente frecuentes.

---

# 13. SGNS: Skip-gram with Negative Sampling

SGNS combina:

Skip-gram + Negative Sampling

Es el algoritmo más usado de Word2Vec.

La función de pérdida es:

L = -log σ(u_o^T v_c) - Σ log σ(-u_n^T v_c)

donde

v_c = embedding palabra central  
u_o = embedding palabra contexto  
u_n = embeddings negativos  
σ = función sigmoide

---

# 14. Intuición geométrica

El entrenamiento empuja los vectores de forma que:

palabras con contextos similares → vectores cercanos.

Esto produce propiedades como:

rey - hombre + mujer ≈ reina

---

# 15. Por qué aparecen analogías

Porque ciertas relaciones semánticas se vuelven **direcciones en el espacio vectorial**.

vector(rey) - vector(hombre)  
≈ vector(reina) - vector(mujer)

---

# 16. Dimensión típica de embeddings

Valores comunes:

50  
100  
200  
300

Trade-off:

dimensión baja → más rápida  
dimensión alta → captura más relaciones

---

# 17. Hiperparámetros importantes

window → tamaño del contexto  
min_count → frecuencia mínima para incluir palabra  
negative → número de muestras negativas (5–20)  
sg = 1 → Skip-gram  
sg = 0 → CBOW

---

# 18. Limitación de Word2Vec

Word2Vec genera **un solo vector por palabra**.

banco (institución financiera)  
banco (asiento)

Ambos comparten el mismo embedding.

Esto se llama **embedding estático**.

---

# 19. Evolución posterior

Modelos posteriores resolvieron esto:

FastText → subpalabras  
ELMo → embeddings contextualizados  
BERT → transformers

---

# 20. Resumen conceptual

Word2Vec funciona porque:

1. usa la hipótesis distribucional  
2. aprende embeddings prediciendo contexto  
3. reduce dimensionalidad  
4. utiliza negative sampling para eficiencia  

Resultado: **un espacio vectorial semántico**
