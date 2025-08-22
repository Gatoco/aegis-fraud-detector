# Análisis de Estructura del Proyecto - API Service

## Situación Actual
```
aegis-fraud-detector/
├── api_service/          # ✅ API FastAPI (Sprint 3.1)
├── src/                  # Código fuente del proyecto ML
├── docs/                 # Documentación y sprints
├── models/               # Artefactos de modelos entrenados
├── data/                 # Datasets y procesamiento
├── notebooks/            # Análisis exploratorio
├── scripts/              # Scripts de entrenamiento
├── tests/                # Tests del proyecto principal
└── ... otros archivos
```

## Ventajas de la Estructura Actual

### ✅ **Pros - Mantener API dentro del proyecto**
1. **Monorepo Coherente**: Todo el ecosistema en un lugar
2. **Referencias Fáciles**: API puede acceder directamente a `../models/`, `../docs/`
3. **Deploy Unificado**: Un solo repositorio para todo
4. **Historial Completo**: Git history del proyecto completo
5. **Compartir Dependencias**: Mismo requirements.txt base

### ❌ **Contras - Mantener API dentro del proyecto**
1. **Acoplamiento**: API ligada al proyecto ML
2. **Deploy Complejo**: Debe incluir todo el proyecto
3. **Escalabilidad**: Difícil separar responsabilidades
4. **Seguridad**: Expone código ML en producción

## Alternativas de Estructura

### Opción A: **API Separada (Recomendado para Producción)**
```
aegis-fraud-detector/      # Proyecto ML
├── src/
├── models/
└── ...

aegis-api-service/         # Repositorio separado
├── main.py
├── models/               # Solo artefactos necesarios
├── docker/
└── deployment/
```

### Opción B: **Mantener Actual (Recomendado para Desarrollo)**
```
aegis-fraud-detector/
├── api_service/          # Mantener aquí
└── ...
```

## Recomendación para AEGIS

### **MANTENER ESTRUCTURA ACTUAL** por:

1. **Estamos en desarrollo**: Sprint 3.1 - fase de construcción
2. **Facilita iteración**: Cambios rápidos entre modelo y API
3. **Testing integrado**: Fácil validación end-to-end
4. **Documentación unificada**: Sprint tracking coherente

### **Migración futura recomendada**:
- **Sprint 4.x**: Separar API para producción
- **Crear CI/CD pipeline** que tome artefactos del proyecto ML
- **API standalone** con solo los componentes necesarios

## Conclusión

Para el contexto actual de AEGIS (desarrollo, sprints, iteración rápida):
**✅ MANTENER API EN `/api_service/`**

La estructura actual es óptima para desarrollo y testing.
La separación se puede hacer más adelante cuando se necesite deploy de producción.
