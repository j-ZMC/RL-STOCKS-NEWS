"""
Script de Validación y Prueba
Verifica formato de datos y hace prueba con pequeño dataset
EJECUTAR ESTO PRIMERO antes de procesar 6M noticias
"""

import pandas as pd
import json
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validar_csv(ruta_csv: str) -> bool:
    """Valida estructura del archivo CSV"""
    logger.info(f"\n{'='*60}")
    logger.info("VALIDANDO CSV DE NOTICIAS")
    logger.info(f"{'='*60}\n")
    
    try:
        # Verificar existencia
        if not os.path.exists(ruta_csv):
            logger.error(f"❌ Archivo no encontrado: {ruta_csv}")
            return False
        
        # Cargar CSV
        df = pd.read_csv(ruta_csv, nrows=100)  # Primeras 100 filas
        
        # Verificar columnas
        columnas_requeridas = ['date', 'headline', 'outlet']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            logger.error(f"❌ Columnas faltantes: {columnas_faltantes}")
            logger.info(f"   Columnas encontradas: {list(df.columns)}")
            return False
        
        logger.info(f"✅ Columnas correctas: {list(df.columns)}")
        
        # Verificar datos
        logger.info(f"\n📊 Análisis de datos:")
        logger.info(f"  Total de filas: {len(pd.read_csv(ruta_csv))}")
        logger.info(f"  Tamaño archivo: {os.path.getsize(ruta_csv) / (1024**3):.2f} GB")
        
        # Verificar valores nulos
        nulos = df.isnull().sum()
        if nulos.any():
            logger.warning(f"⚠️  Valores nulos detectados:")
            for col, count in nulos[nulos > 0].items():
                logger.warning(f"   {col}: {count} nulos ({count/len(df)*100:.1f}%)")
        else:
            logger.info(f"✅ Sin valores nulos")
        
        # Verificar titulares
        titulares_vacios = (df['headline'].str.strip() == '').sum()
        if titulares_vacios > 0:
            logger.warning(f"⚠️  {titulares_vacios} titulares vacíos")
        else:
            logger.info(f"✅ Todos los titulares tienen contenido")
        
        # Longitud promedio de titulares
        longitud_promedio = df['headline'].str.len().mean()
        logger.info(f"   Longitud promedio de titular: {longitud_promedio:.0f} caracteres")
        
        # Outlets únicos
        outlets_unicos = df['outlet'].nunique()
        logger.info(f"   Outlets únicos: {outlets_unicos}")
        
        logger.info(f"\n✅ CSV válido para procesamiento\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validando CSV: {e}")
        return False


def validar_taxonomia(ruta_json: str) -> bool:
    """Valida estructura del archivo JSON de taxonomía"""
    logger.info(f"{'='*60}")
    logger.info("VALIDANDO TAXONOMÍA CGIC")
    logger.info(f"{'='*60}\n")
    
    try:
        # Verificar existencia
        if not os.path.exists(ruta_json):
            logger.error(f"❌ Archivo no encontrado: {ruta_json}")
            return False
        
        # Cargar JSON
        with open(ruta_json, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        
        # Verificar estructura principal
        if 'industrias' not in taxonomy:
            logger.error("❌ JSON debe tener clave 'industrias'")
            return False
        
        logger.info(f"✅ Estructura JSON válida")
        
        # Análisis
        industrias = taxonomy['industrias']
        logger.info(f"\n📊 Análisis de Taxonomía:")
        logger.info(f"  Total de industrias: {len(industrias)}")
        
        total_subindustrias = 0
        for ind in industrias:
            if 'id' not in ind or 'nombre' not in ind:
                logger.error(f"❌ Industria sin id o nombre: {ind}")
                return False
            
            subinds = ind.get('subindustrias', [])
            total_subindustrias += len(subinds)
            
            for subind in subinds:
                if 'id' not in subind or 'nombre' not in subind:
                    logger.error(f"❌ Sub-industria sin id o nombre: {subind}")
                    return False
        
        logger.info(f"  Total de sub-industrias: {total_subindustrias}")
        
        # Verificar IDs únicos
        ids_industrias = [ind['id'] for ind in industrias]
        if len(ids_industrias) != len(set(ids_industrias)):
            logger.error("❌ IDs de industrias duplicados")
            return False
        
        logger.info(f"✅ Todos los IDs de industrias son únicos")
        
        # Listar industrias
        logger.info(f"\n📋 Industrias CGIC:")
        for ind in industrias:
            subinds = ind.get('subindustrias', [])
            logger.info(f"  [{ind['id']}] {ind['nombre']} ({len(subinds)} sub-industrias)")
        
        logger.info(f"\n✅ Taxonomía válida para procesamiento\n")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error en formato JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error validando taxonomía: {e}")
        return False


def prueba_embeddings():
    """Prueba rápida de embeddings con noticias de ejemplo"""
    logger.info(f"{'='*60}")
    logger.info("PRUEBA DE EMBEDDINGS (TEST RÁPIDO)")
    logger.info(f"{'='*60}\n")
    
    try:
        logger.info("Instalando/verificando librerías requeridas...")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            logger.error("❌ Librerías no instaladas. Ejecutar:")
            logger.error("   pip install sentence-transformers torch")
            return False
        
        logger.info(f"✅ Librerías disponibles")
        logger.info(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        
        # Cargar modelo
        logger.info("\nCargando modelo de embeddings...")
        model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        logger.info(f"✅ Modelo cargado exitosamente")
        logger.info(f"   Dimensión de embedding: {model.get_sentence_embedding_dimension()}")
        
        # Prueba con noticias de ejemplo
        titulares_prueba = [
            "Tesla anuncia inversión de $5 mil millones en nueva planta de baterías",
            "Banco Central mantiene tasa de interés en 4.75%",
            "Microsoft adquiere startup de inteligencia artificial"
        ]
        
        logger.info(f"\nEncodificando {len(titulares_prueba)} titulares de prueba...")
        embeddings = model.encode(titulares_prueba, show_progress_bar=False)
        logger.info(f"✅ Embeddings generados")
        logger.info(f"   Shape: {embeddings.shape}")
        
        # Calcular similitud entre titulares
        from sklearn.metrics.pairwise import cosine_similarity
        similitudes = cosine_similarity(embeddings)
        
        logger.info(f"\n📊 Similitud entre titulares:")
        for i in range(len(titulares_prueba)):
            for j in range(i+1, len(titulares_prueba)):
                sim = similitudes[i][j]
                logger.info(f"   Titular {i+1} vs {j+1}: {sim:.3f}")
        
        logger.info(f"\n✅ Sistema de embeddings funciona correctamente\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en prueba de embeddings: {e}")
        return False


def main():
    """Ejecutar todas las validaciones"""
    
    logger.info("\n" + "="*60)
    logger.info("VALIDADOR Y PROBADOR - CLASIFICADOR DE NOTICIAS CGIC")
    logger.info("="*60 + "\n")
    
    # Archivos a validar
    ruta_csv = "ejemplo_noticias.csv"
    ruta_json = "ejemplo_cgic_taxonomy.json"
    
    logger.info("Usando archivos de ejemplo para demostración.\n")
    logger.info("Cuando uses TUS datos, cambiar en línea ~220:\n")
    logger.info('  ruta_csv = "tu_archivo_noticias.csv"')
    logger.info('  ruta_json = "tu_taxonomia_cgic.json"\n')
    
    # Validaciones
    validaciones_ok = [
        validar_csv(ruta_csv),
        validar_taxonomia(ruta_json),
        prueba_embeddings()
    ]
    
    # Resumen
    logger.info("="*60)
    if all(validaciones_ok):
        logger.info("✅ TODAS LAS VALIDACIONES PASARON")
        logger.info("="*60 + "\n")
        logger.info("✅ Sistema listo para procesar 6 millones de noticias\n")
        logger.info("Próximos pasos:")
        logger.info("1. Reemplaza rutas en noticias_classifier_faiss.py")
        logger.info("2. Ajusta BATCH_SIZE según tu RAM disponible")
        logger.info("3. Ejecuta: python noticias_classifier_faiss.py\n")
        return True
    else:
        logger.info("❌ ALGUNAS VALIDACIONES FALLARON")
        logger.info("="*60 + "\n")
        logger.info("Revisa los errores arriba y corrige antes de continuar.\n")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
