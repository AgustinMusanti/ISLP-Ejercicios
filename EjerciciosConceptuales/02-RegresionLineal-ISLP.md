# Capítulo 3 — Regresión Lineal

La idea de fondo es predecir una variable numérica **Y** (ej. ventas, balance) como una **suma ponderada** de variables explicativas **X**. El modelo es una recta (o un plano, si hay varias X):

> Y ≈ β₀ + β₁·X₁ + β₂·X₂ + … + βₚ·Xₚ

Los **β** (coeficientes) son los números que el modelo aprende de los datos.

---

## 1. Regresión lineal simple (una sola X)

> Y ≈ β₀ + β₁·X

- **β₀ (intercept)**: valor de Y cuando X = 0. Es el punto de partida de la recta.
- **β₁ (pendiente)**: cuánto cambia Y por cada unidad que sube X.

**Ejemplo:** `ventas ≈ 7 + 0.05·TV`. Cada $1 más en publicidad de TV suma 0.05 unidades de venta; sin TV, las ventas base son 7.

### ¿Cómo encuentra la recta? → Mínimos cuadrados (RSS)

El modelo busca la recta que **minimiza el error total**. Para cada punto calcula el *residuo* (valor real − valor predicho), lo eleva al cuadrado y los suma todos:

- **RSS (Residual Sum of Squares)** = suma de los errores al cuadrado. La recta "ganadora" es la que da el RSS más chico.

### ¿Puedo confiar en los coeficientes?

Como los β se estiman con una muestra, no son exactos. Cuatro números te dicen qué tan firmes son:

- **Std. error (SE)**: la incertidumbre del coeficiente. Más chico = más preciso.
- **t-statistic** = coeficiente / SE. Mide a cuántos SE de distancia está el coeficiente del cero. Grande = lejos del cero = la variable parece importar.
- **p-value**: probabilidad de ver este resultado si la variable **no** tuviera efecto. **< 0.05 → la variable es significativa**, descartás que sea casualidad.
- **Intervalo de confianza**: el rango donde probablemente cae el coeficiente real (ej. β₁ entre 0.04 y 0.06).

### ¿Qué tan bueno es el modelo?

- **RSE (Residual Standard Error)**: error típico del modelo, en las **mismas unidades** que Y. Un RSE de 1.69 = en promedio le erro por ~1.69 unidades. Más chico, mejor.
- **R²**: proporción de la variabilidad de Y que el modelo explica. Va de 0 a 1. **R² = 0.61 → explica el 61%**. No depende de unidades, por eso es la métrica más usada para resumir el ajuste.

---

## 2. Regresión lineal múltiple (varias X)

Misma idea, pero con varios predictores a la vez:

> Y ≈ β₀ + β₁·X₁ + … + βₚ·Xₚ

Clave en la interpretación: cada coeficiente es el efecto de **esa** variable **manteniendo las demás fijas**. (Ojo: una variable que se veía importante sola puede dejar de serlo al sumar otras correlacionadas).

Al pasar a múltiple, hay **4 preguntas** que resolver:

### a) ¿Sirve al menos una variable? → F-statistic

Chequeo **global** del modelo. Responde sí/no: *¿al menos un predictor tiene relación real con Y, o es todo ruido?*

- **F grande** (ej. 570, muy lejos de 1) → sí, el modelo en conjunto sirve.
- **F ≈ 1** → las variables no aportan nada.

Se mira **antes** que los coeficientes individuales.

### b) ¿Cuáles variables importan? → selección de variables

No conviene quedarse con todas. Estrategias para elegir un subconjunto:

- **Forward selection**: arrancás sin variables y vas agregando la que más aporta.
- **Backward selection**: arrancás con todas y vas sacando la menos útil.
- **Mixed**: combinación de las dos.

### c) ¿Qué tan bien ajusta? → RSE y R² (igual que antes)

Advertencia importante: **agregar variables casi siempre sube el R²**, aunque sean inútiles. Por eso se usa el **R² ajustado**, que penaliza meter variables de más. Para comparar modelos con distinta cantidad de variables, mirá el ajustado.

### d) Predicciones → dos tipos de intervalo

- **Intervalo de confianza**: incertidumbre sobre el promedio (ej. ventas promedio de *todas* las ciudades con este presupuesto).
- **Intervalo de predicción**: incertidumbre sobre *un caso puntual*. Siempre **más ancho**, porque suma el error individual.

---

## 3. Otras consideraciones

### Variables cualitativas (categorías) → dummies

Si X es categórica (ej. "estudiante: sí/no"), se convierte en una variable **0/1** (dummy). El coeficiente mide la diferencia respecto a la categoría base. Con más de 2 categorías (ej. región: norte/sur/este) se crean varias dummies, una menos que la cantidad de categorías.

### Interacciones (efecto sinergia)

A veces dos variables **juntas** hacen más que por separado. Se agrega un término `X₁·X₂`.

**Ejemplo:** gastar en TV **y** radio a la vez rinde más que la suma de cada una por su lado. Esa "sinergia" la captura el término de interacción.

### No linealidad → regresión polinómica

Si la relación es curva, agregás potencias: `Y ≈ β₀ + β₁·X + β₂·X²`. Sigue siendo regresión lineal (es lineal en los β), solo que ahora ajusta una curva.

### 6 problemas a vigilar

1. **No linealidad**: la relación real no es recta. Se detecta con un *gráfico de residuos* (si quedan con forma/patrón, algo falta).
2. **Errores correlacionados**: típico en datos de tiempo; viola un supuesto y subestima los errores.
3. **Varianza no constante (heterocedasticidad)**: el error crece con Y (residuos en forma de embudo). Se suele arreglar transformando Y (ej. log).
4. **Outliers**: puntos con Y muy lejos de lo predicho. Inflan el RSE.
5. **High leverage**: puntos con un X extremo/raro que tiran de la recta con fuerza desmedida.
6. **Colinealidad**: dos predictores muy correlacionados entre sí (ej. límite y saldo de tarjeta). Confunde al modelo y hace inestables los coeficientes. Se mide con el **VIF (Variance Inflation Factor)**: VIF > 5–10 = problema; conviene sacar una de las variables.

---

## 4. Regresión lineal vs. KNN (paramétrico vs. no paramétrico)

- **Regresión lineal** (paramétrico): asume forma de recta. Si la relación **es** aprox. lineal, gana: simple, interpretable, necesita pocos datos.
- **KNN** (no paramétrico): no asume forma, se adapta a cualquier relación. Gana cuando la relación es **muy no lineal**.

Pero KNN sufre la **maldición de la dimensionalidad**: con muchas variables, los "vecinos" dejan de estar cerca y el método empeora. Regla práctica: con muchas dimensiones y/o pocos datos, **la regresión lineal suele rendir mejor** aunque la relación no sea perfectamente lineal.

---

### Glosario express

| Sigla | Qué es | Para qué sirve |
|---|---|---|
| **RSS** | Suma de errores al cuadrado | Lo que el modelo minimiza para encontrar la recta |
| **RSE** | Error típico (en unidades de Y) | Saber cuánto le erro en promedio |
| **R²** | % de variabilidad explicada (0–1) | Resumir qué tan bien ajusta |
| **F-stat** | Test global del modelo | ¿Sirve al menos una variable? |
| **t-stat** | Coef. / SE | Cuán lejos del cero está un coeficiente |
| **p-value** | Prob. de que sea casualidad | < 0.05 → la variable importa |
| **VIF** | Factor de inflación de varianza | Detectar colinealidad (>5–10 = problema) |
