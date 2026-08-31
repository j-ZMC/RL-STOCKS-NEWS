"""
Clasificador de Noticias - Versión FAISS Optimizada
Máxima precisión con búsqueda vectorial ultra-rápida
Recomendado para datasets grandes (6M+ registros)
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
import faiss
import logging
from tqdm import tqdm
import gc
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CGICNewsClassifierFAISS:
    """Clasificador con índice FAISS para máxima velocidad y precisión"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        confidence_threshold: float = 0.55,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Cargando modelo en {device.upper()}...")
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.taxonomy = {}
        
        # Índices FAISS
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.subindustrias_mapping = []  # Mapeo de índice FAISS a ID
        
    def cargar_taxonomia(self, ruta_json: str) -> None:
        """Carga taxonomía y construye índice FAISS"""
        logger.info(f"Cargando taxonomía desde: {ruta_json}")
        
        with open(ruta_json, 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)

        if 'industrias' not in self.taxonomy and 'categorias' in self.taxonomy:
            categorias_por_sector = self._agrupar_categorias_por_sector()
            self.taxonomy = {
                'industrias': [
                    {
                        'id': sector,
                        'subindustrias': [
                            {
                                'id': categoria['id'],
                                'nombre': categoria['categoria'],
                            }
                            for categoria in categorias
                        ]
                    }
                    for sector, categorias in categorias_por_sector.items()
                ]
            }
        
        logger.info("Construyendo índice FAISS...")
        self._construir_indice_faiss()
        
    def _construir_indice_faiss(self) -> None:
        """Construye índice FAISS con todas las sub-industrias"""
        
        embeddings_lista = []
        
        for industria in self.taxonomy.get('industrias', []):
            for subind in industria.get('subindustrias', []):
                # Generar embedding
                embedding = self.model.encode(
                    subind['nombre'],
                    convert_to_tensor=False,
                    device=self.device
                ).astype(np.float32)
                
                embeddings_lista.append(embedding)
                
                # Mapeo para recuperar información
                self.subindustrias_mapping.append({
                    'id': subind['id'],
                    'nombre': subind['nombre'],
                    'industria_padre': industria['id']
                })
        
        # Construir índice FAISS
        embeddings_array = np.array(embeddings_lista)
        logger.info(f"Agregando {len(embeddings_array)} vectores al índice...")
        self.faiss_index.add(embeddings_array)
        
        logger.info(f"Índice construido: {self.faiss_index.ntotal} sub-industrias indexadas")

    def _agrupar_categorias_por_sector(self) -> Dict[str, List[Dict]]:
        agrupadas = {}
        for categoria in self.taxonomy.get('categorias', []):
            sector = categoria.get('sector', 'sin_sector')
            agrupadas.setdefault(sector, []).append(categoria)
        return agrupadas
    
    def clasificar_noticia(
        self,
        titular: str,
        top_k: int = 3
    ) -> Dict:
        """Clasifica una noticia usando búsqueda FAISS"""
        
        # Generar embedding
        embedding = self.model.encode(
            titular,
            convert_to_tensor=False
        ).astype(np.float32)
        
        # Búsqueda en FAISS
        distances, indices = self.faiss_index.search(
            np.array([embedding]),
            top_k
        )
        
        # Convertir distancias L2 a similitud coseno
        # Distancia L2: d = sqrt(2 - 2*cosine_sim)
        # Por lo que: cosine_sim = 1 - (d^2 / 2)
        classificaciones = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 significa no encontrado
                dist = distances[0][i]
                # Convertir L2 distance a cosine similarity
                similitud = 1 - dist / 2
                
                subind_info = self.subindustrias_mapping[idx]
                
                classificaciones.append({
                    'subindustria_id': subind_info['id'],
                    'subindustria_nombre': subind_info['nombre'],
                    'industria_id': subind_info['industria_padre'],
                    'score': float(similitud)
                })
        
        resultado = {
            'titular': titular,
            'clasificaciones': classificaciones,
            'confianza_maxima': classificaciones[0]['score'] if classificaciones else 0.0,
            'clasificado': classificaciones[0]['score'] >= self.confidence_threshold if classificaciones else False
        }
        
        return resultado
    
    def procesar_csv_batch(
        self,
        ruta_csv: str,
        ruta_salida: str,
        batch_size: int = 128,
        columna_titular: str = 'headline',
        top_k: int = 1
    ) -> Tuple[int, int]:
        """
        Procesa CSV en batches con máxima eficiencia
        
        Returns:
            Tupla (total_procesados, total_clasificados)
        """
        logger.info(f"Cargando CSV: {ruta_csv}")
        df = pd.read_csv(ruta_csv, low_memory=False)
        total_registros = len(df)
        
        logger.info(f"Total de registros: {total_registros}")
        logger.info(f"Batch size: {batch_size}")
        
        resultados = []
        clasificados = 0
        
        inicio = time.time()
        
        # Procesar en batches grandes
        for batch_start in tqdm(
            range(0, total_registros, batch_size),
            desc="Procesando noticias con FAISS"
        ):
            batch_end = min(batch_start + batch_size, total_registros)
            batch = df.iloc[batch_start:batch_end]
            
            # Generar embeddings para todo el batch
            titulares = batch[columna_titular].fillna('').tolist()
            
            try:
                embeddings_batch = self.model.encode(
                    titulares,
                    convert_to_tensor=False,
                    batch_size=batch_size,
                    device=self.device
                ).astype(np.float32)
                
                # Búsqueda FAISS para todo el batch
                distances, indices = self.faiss_index.search(
                    embeddings_batch,
                    top_k
                )
                
                # Conservar todas las sub-industrias que superen el umbral.
                for idx_en_batch, (_, row) in enumerate(batch.iterrows()):
                    for rank in range(indices.shape[1]):
                        dist = distances[idx_en_batch][rank]
                        idx_subind = indices[idx_en_batch][rank]

                        if idx_subind == -1:
                            continue

                        # FAISS IndexFlatL2 returns squared L2 distance. For
                        # normalized embeddings, cosine similarity is 1 - d / 2.
                        similitud = 1 - dist / 2

                        if similitud < self.confidence_threshold:
                            continue

                        subind_info = self.subindustrias_mapping[idx_subind]
                        resultado_row = row.to_dict()
                        resultado_row.update({
                            'industria_id': subind_info['industria_padre'],
                            'subindustria_id': subind_info['id'],
                            'subindustria_nombre': subind_info['nombre'],
                            'confidence_score': float(similitud)
                        })
                        resultados.append(resultado_row)
                        clasificados += 1
                
                # Limpiar memoria
                del embeddings_batch
                
            except Exception as e:
                logger.error(f"Error procesando batch {batch_start}-{batch_end}: {e}")
                continue
            
            # Liberar memoria cada 10 batches
            if batch_start % (batch_size * 10) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Guardar resultados
        columnas_salida = list(df.columns) + [
            'industria_id',
            'subindustria_id',
            'subindustria_nombre',
            'confidence_score'
        ]
        df_resultados = pd.DataFrame(resultados, columns=columnas_salida)
        df_resultados.to_csv(ruta_salida, index=False, encoding='utf-8')
        
        tiempo_total = time.time() - inicio
        velocidad = total_registros / tiempo_total
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ PROCESAMIENTO COMPLETADO")
        logger.info(f"{'='*60}")
        logger.info(f"Tiempo total: {tiempo_total:.2f} segundos")
        logger.info(f"Velocidad: {velocidad:.0f} noticias/segundo")
        logger.info(f"Total registros: {total_registros}")
        logger.info(f"Clasificados: {clasificados} ({clasificados/total_registros*100:.2f}%)")
        logger.info(f"Resultados guardados: {ruta_salida}")
        logger.info(f"{'='*60}\n")
        
        return total_registros, clasificados
    
    def validar_clasificaciones(
        self,
        ruta_csv_clasificado: str,
        muestra_size: int = 100
    ) -> Dict:
        """Valida la calidad de clasificaciones (muestreo)"""
        
        logger.info(f"Validando clasificaciones (muestra de {muestra_size})...")
        
        df = pd.read_csv(ruta_csv_clasificado)
        
        if len(df) < muestra_size:
            muestra = df
        else:
            muestra = df.sample(n=muestra_size, random_state=42)
        
        estadisticas = {
            'total_muestreado': len(muestra),
            'confianza_promedio': float(muestra['confidence_score'].mean()),
            'confianza_min': float(muestra['confidence_score'].min()),
            'confianza_max': float(muestra['confidence_score'].max()),
            'desv_est': float(muestra['confidence_score'].std()),
            'industrias_unicas': muestra['industria_id'].nunique(),
            'subindustrias_unicas': muestra['subindustria_id'].nunique()
        }
        
        # Distribución por industria
        dist_industrias = muestra['industria_id'].value_counts().to_dict()
        estadisticas['distribucion_industrias'] = dist_industrias
        
        logger.info(f"\nEstadísticas de validación:")
        logger.info(f"  Confianza promedio: {estadisticas['confianza_promedio']:.3f}")
        logger.info(f"  Rango: [{estadisticas['confianza_min']:.3f}, {estadisticas['confianza_max']:.3f}]")
        logger.info(f"  Industrias únicas: {estadisticas['industrias_unicas']}")
        
        return estadisticas


def main():
    """Script principal"""
    
    # ========== CONFIGURACIÓN ==========
    RUTA_CSV_NOTICIAS = "noticias.csv"
    RUTA_JSON_TAXONOMIA = "cgic_taxonomy.json"
    RUTA_SALIDA = "noticias_clasificadas.csv"
    BATCH_SIZE = 128  # Aumentar si tienes mucha RAM (256, 512)
    CONFIDENCE_THRESHOLD = 0.55  # Aumentar para más precisión, bajar para más cobertura
    
    # ========== PROCESAMIENTO ==========
    clasificador = CGICNewsClassifierFAISS(
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    clasificador.cargar_taxonomia(RUTA_JSON_TAXONOMIA)
    
    total, clasificados = clasificador.procesar_csv_batch(
        RUTA_CSV_NOTICIAS,
        RUTA_SALIDA,
        batch_size=BATCH_SIZE,
        columna_titular='headline'
    )
    
    # Validación
    clasificador.validar_clasificaciones(RUTA_SALIDA, muestra_size=100)


if __name__ == "__main__":
    main()
