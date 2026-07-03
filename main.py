from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Inicialización de la aplicación
app = FastAPI(
    title="API de Predicción de Abandono de Clientes",
    description="Motor de Machine Learning para calcular el riesgo de deserción en tiempo real.",
    version="1.0.0"
)

# Configuración de CORS (CRÍTICO para que WordPress o tu frontend web pueda comunicarse con la API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambia "*" por la URL exacta de tu web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga de modelos al iniciar el servidor (evita recargar los archivos en cada petición)
try:
    modelo = joblib.load('modelo_abandono.joblib')
    columnas = joblib.load('columnas_modelo.joblib')
    print(" Modelos de IA cargados en memoria exitosamente.")
except Exception as e:
    print(f" Error al cargar el modelo. Asegúrate de tener los archivos .joblib. Detalles: {e}")

# Definición estricta de los datos de entrada usando Pydantic
class ClienteInput(BaseModel):
    edad: int = Field(..., gt=0, description="Edad del cliente")
    ciudad: str = Field(..., description="Ciudad de residencia")
    tipo_cliente: str = Field(..., description="Categoría del cliente")
    dias_registrado: int = Field(..., ge=0)
    compras_totales: int = Field(..., ge=0)
    compras_ultimos_90_dias: int = Field(..., ge=0)
    gasto_total_90_dias: float = Field(..., ge=0.0)
    promedio_compra: float = Field(..., ge=0.0)
    dias_desde_ultima_compra: int = Field(..., ge=0)
    frecuencia_compra_dias: int = Field(..., ge=0)
    regularidad_compra: float = Field(..., ge=0.0)
    porcentaje_descuentos_usados: float = Field(..., ge=0.0, le=1.0)
    usa_domicilio: int = Field(..., ge=0, le=1)
    reclamos: int = Field(..., ge=0)
    calificacion_promedio: float = Field(..., ge=0.0, le=5.0)

# Endpoint de prueba (Health Check)
@app.get("/")
def health_check():
    return {"estado": "API funcionando correctamente, lista para recibir peticiones."}

# Endpoint principal de predicción
@app.post("/api/v1/predecir", summary="Calcula el riesgo de abandono de un usuario")
def predecir_abandono(cliente: ClienteInput):
    try:
        # 1. Convertir el JSON recibido a un DataFrame de Pandas
        df_nuevo = pd.DataFrame([cliente.model_dump()])
        
        # 2. Aplicar One-Hot Encoding (igual que en tu notebook de Jupyter)
        df_nuevo_prepared = pd.get_dummies(df_nuevo)
        
        # 3. Alinear columnas: rellenar con 0 las que falten y mantener el orden exacto
        for col in columnas:
            if col not in df_nuevo_prepared.columns:
                df_nuevo_prepared[col] = 0
                
        df_final = df_nuevo_prepared[columnas]
        
        # 4. Realizar la predicción matemática
        prediccion = modelo.predict(df_final)[0]
        probabilidades = modelo.predict_proba(df_final)[0]
        
        # 5. Estructurar una respuesta enriquecida para el frontend
        probabilidad_abandono = float(probabilidades[1])
        
        return {
            "exito": True,
            "prediccion": {
                "riesgo_abandono": bool(prediccion == 1),
                "probabilidad_porcentaje": round(probabilidad_abandono * 100, 2),
                "nivel_riesgo": "Alto" if probabilidad_abandono > 0.70 else ("Medio" if probabilidad_abandono > 0.40 else "Bajo")
            },
            "mensaje": "Análisis completado con éxito."
        }
        
    except Exception as e:
        # Manejo de errores para evitar que un dato corrupto tumbe el servidor
        raise HTTPException(status_code=500, detail=f"Error interno procesando la predicción: {str(e)}")