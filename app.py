import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from contextlib import asynccontextmanager
import mlflow
from mlflow.models import infer_signature



mlflow.set_tracking_uri("databricks")

RUTA_MODELO = "models:/workspace.default.modelofinal/1"


# CARGA DEL MODELO
MODEL_PATH = "modelo_final.pkl"
storage = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        storage["model"] = mlflow.pyfunc.load_model(RUTA_MODELO)
        print("Modelo PKL cargado correctamente.")
    except Exception as e:
        print(f"Error crítico cargando modelo: {e}")
        storage["model"] = None
    yield
    storage.clear()

app = FastAPI(title="API Predicción Canarias", lifespan=lifespan)

# Datos que puede recibir mi API 
class ParadosInput(BaseModel):
    AÑO: int
    MES: int = Field(..., ge=1, le=12)
    I_El_Hierro: bool = Field(False, alias="I_El Hierro")
    I_Fuerteventura: bool = Field(False, alias="I_Fuerteventura")
    I_Gran_Canaria: bool = Field(False, alias="I_Gran Canaria")
    I_La_Gomera: bool = Field(False, alias="I_La Gomera")
    I_La_Palma: bool = Field(False, alias="I_La Palma") 
    I_Lanzarote: bool = Field(False, alias="I_Lanzarote")
    I_Tenerife: bool = Field(False, alias="I_Tenerife")
    E_25_años_o_más: bool = Field(False, alias="E_25 años o más")
    E_Menor_de_25_años: bool = Field(False, alias="E_Menor de 25 años")
    N_V_Educación_general: bool = Field(False, alias="N_V_Educación general")
    N_V_Estudios_post_secundarios: bool = Field(False, alias="N_V_Estudios post-secundarios")
    N_V_Estudios_primarios: bool = Field(False, alias="N_V_Estudios primarios")
    N_V_Estudios_primarios_completos: bool = Field(False, alias="N_V_Estudios primarios completos")
    N_V_Estudios_primarios_incompletos: bool = Field(False, alias="N_V_Estudios primarios incompletos")
    N_V_Estudios_secundarios: bool = Field(False, alias="N_V_Estudios secundarios")
    N_V_Formación_profesional: bool = Field(False, alias="N_V_Formación profesional")
    N_V_Primer_ciclo: bool = Field(False, alias="N_V_Primer ciclo")
    N_V_Resto_de_estudios_post_secundarios: bool = Field(False, alias="N_V_Resto de estudios post-secundarios")
    N_V_Segundo_y_tercer_ciclo: bool = Field(False, alias="N_V_Segundo y tercer ciclo")
    N_V_Sin_estudios: bool = Field(False, alias="N_V_Sin estudios")
    N_V_Técnico_profesional_superior: bool = Field(False, alias="N_V_Técnico profesional superior")
    PIB_TOTAL: float
    CRISIS: bool
    COVID: bool

    class Config:
        populate_by_name = True
    
    # Validación de datos
    @model_validator(mode='after')
    def validar_reglas_negocio(self):
        # Validar Islas 
        islas = [self.I_El_Hierro, self.I_Fuerteventura, self.I_Gran_Canaria, 
                    self.I_La_Gomera, self.I_La_Palma, self.I_Lanzarote, self.I_Tenerife]
        if sum(islas) > 1:
            raise ValueError("No puedes seleccionar más de una isla a la vez.")
        
        # Validar Edad 
        edades = [self.E_25_años_o_más, self.E_Menor_de_25_años]
        if sum(edades) > 1:
            raise ValueError("No puedes seleccionar ambos rangos de edad al mismo tiempo.")

        # Validar Nivel de Estudios (N_V) 
        estudios = [
            self.N_V_Educación_general, self.N_V_Estudios_post_secundarios, 
            self.N_V_Estudios_primarios, self.N_V_Estudios_primarios_completos,
            self.N_V_Estudios_primarios_incompletos, self.N_V_Estudios_secundarios,
            self.N_V_Formación_profesional, self.N_V_Primer_ciclo,
            self.N_V_Resto_de_estudios_post_secundarios, self.N_V_Segundo_y_tercer_ciclo,
            self.N_V_Sin_estudios, self.N_V_Técnico_profesional_superior
        ]
        if sum(estudios) > 1:
            raise ValueError("No puedes seleccionar más de un nivel de estudios (N_V) a la vez.")

        return self

# Endpoints

@app.get("/health")
def health():
    return {"status": "ok" if storage.get("model") else "ko"}

@app.post("/predict")
def predict(data: ParadosInput):
    model = storage.get("model")
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        # Convertir a DataFrame usando los nombres de los ALIAS 
        df_input = pd.DataFrame([data.model_dump(by_alias=True)])
        
        # Aseguramos que el DF tenga el orden correcto
        storage["columns"] = joblib.load("columnas.pkl")

        df_input = df_input[storage["columns"]]
        
        pred = model.predict(df_input)
        return {"prediction": float(pred[0])}
        
    except Exception as e:
        # Aparece si alguna columna en ParadosInput o el alias está mal
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
    
    
