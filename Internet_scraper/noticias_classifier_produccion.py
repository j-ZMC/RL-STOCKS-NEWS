"""
================================================================================
CLASIFICADOR DE NOTICIAS CGIC - VERSIÓN FINAL OPTIMIZADA
================================================================================
• Autodetecta VRAM y configura automáticamente
• Batch size adaptativo (32-2048)
• Modelo automático según GPU disponible
• Máxima velocidad y precisión
• Limpeza de memoria optimizada
• IndexIVFFlat para búsquedas ultra-rápidas
================================================================================
"""

import pandas as pd
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer
import faiss
import logging
from tqdm import tqdm
import gc
import time
from pathlib import Path
import psutil

# ==================== CONFIGURACIÓN DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== DETECCIÓN AUTOMÁTICA DE HARDWARE ====================
def detectar_hardware():
    """
    Detecta GPU disponible y recursos del sistema
    Retorna configuración óptima
    """
    logger.info("\n" + "="*80)
    logger.info("🔍 DETECTANDO HARDWARE Y CONFIGURACIÓN ÓPTIMA")
    logger.info("="*80 + "\n")
    
    config = {
        'device': 'cpu',
        'vram_total': 0,
        'vram_usable': 0,
        'batch_size': 32,
        'model_name': 'paraphrase-multilingual-mpnet-base-v2',
        'clean_frequency': 10,
        'use_ivf': False
    }
    
    # Detectar GPU
    if torch.cuda.is_available():
        config['device'] = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Restar 1GB de seguridad
        vram_usable = vram_total - 1.0
        
        config['vram_total'] = vram_total
        config['vram_usable'] = vram_usable
        
        logger.info(f"✅ GPU DETECTADA: {gpu_name}")
        logger.info(f"   VRAM Total: {vram_total:.1f} GB")
        logger.info(f"   VRAM Usable (menos 1GB buffer): {vram_usable:.1f} GB\n")
        
        # Configurar automáticamente según VRAM
        if vram_usable >= 20:
            config['batch_size'] = 2048
            config['model_name'] = 'xlm-r-large-v2'
            config['clean_frequency'] = 100
            config['use_ivf'] = True
            logger.info("⚙️  CONFIGURACIÓN: MÁXIMA (24GB+ VRAM)")
            logger.info("   • Batch Size: 2048")
            logger.info("   • Modelo: XLM-R Large (máxima precisión)")
            logger.info("   • Limpiar memoria: cada 100 batches")
            logger.info("   • Index: IVFFlat\n")
            
        elif vram_usable >= 14:
            config['batch_size'] = 1024
            config['model_name'] = 'distiluse-base-multilingual-cased-v2'
            config['clean_frequency'] = 50
            config['use_ivf'] = True
            logger.info("⚙️  CONFIGURACIÓN: ALTA (14-24GB VRAM)")
            logger.info("   • Batch Size: 1024")
            logger.info("   • Modelo: DistilUSE (balance precision/velocidad)")
            logger.info("   • Limpiar memoria: cada 50 batches")
            logger.info("   • Index: IVFFlat\n")
            
        elif vram_usable >= 10:
            config['batch_size'] = 512
            config['model_name'] = 'distiluse-base-multilingual-cased-v2'
            config['clean_frequency'] = 30
            config['use_ivf'] = True
            logger.info("⚙️  CONFIGURACIÓN: MEDIA-ALTA (10-14GB VRAM)")
            logger.info("   • Batch Size: 512")
            logger.info("   • Modelo: DistilUSE")
            logger.info("   • Limpiar memoria: cada 30 batches")
            logger.info("   • Index: IVFFlat\n")
            
        elif vram_usable >= 6:
            config['batch_size'] = 256
            config['model_name'] = 'paraphrase-multilingual-mpnet-base-v2'
            config['clean_frequency'] = 20
            config['use_ivf'] = False
            logger.info("⚙️  CONFIGURACIÓN: MEDIA (6-10GB VRAM)")
            logger.info("   • Batch Size: 256")
            logger.info("   • Modelo: Multilingual MPN (estándar)")
            logger.info("   • Limpiar memoria: cada 20 batches")
            logger.info("   • Index: FlatL2\n")
            
        elif vram_usable >= 4:
            config['batch_size'] = 128
            config['model_name'] = 'paraphrase-multilingual-mpnet-base-v2'
            config['clean_frequency'] = 10
            config['use_ivf'] = False
            logger.info("⚙️  CONFIGURACIÓN: MEDIA-BAJA (4-6GB VRAM)")
            logger.info("   • Batch Size: 128")
            logger.info("   • Modelo: Multilingual MPN")
            logger.info("   • Limpiar memoria: cada 10 batches")
            logger.info("   • Index: FlatL2\n")
            
        else:
            config['batch_size'] = 64
            config['model_name'] = 'paraphrase-multilingual-mpnet-base-v2'
            config['clean_frequency'] = 5
            config['use_ivf'] = False
            logger.info("⚙️  CONFIGURACIÓN: BÁSICA (<4GB VRAM)")
            logger.info("   • Batch Size: 64")
            logger.info("   • Modelo: Multilingual MPN")
            logger.info("   • Limpiar memoria: cada 5 batches")
            logger.info("   • Index: FlatL2\n")
    else:
        logger.warning("⚠️  GPU NO DISPONIBLE - Usando CPU")
        logger.warning("   Para máxima velocidad, instala CUDA:")
        logger.warning("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        logger.warning("   pip install faiss-gpu\n")
        
        config['device'] = 'cpu'
        config['batch_size'] = 32
        config['model_name'] = 'paraphrase-multilingual-mpnet-base-v2'
        logger.info("⚙️  CONFIGURACIÓN: CPU MODE")
        logger.info("   • Batch Size: 32")
        logger.info("   • Modelo: Multilingual MPN")
        logger.info("   • Nota: Será más lento que GPU\n")
    
    return config


# ==================== CLASE PRINCIPAL DEL CLASIFICADOR ====================
class ClasificadorCGICFinal:
    """
    Clasificador FAISS de noticias - Versión final optimizada
    
    Características:
    - Autodetección de hardware
    - Batch size adaptativo
    - Modelo automático
    - Índice FAISS optimizado (IVFFlat o FlatL2)
    - Logging detallado
    - Estadísticas en tiempo real
    """
    
    def __init__(self, config: Dict = None):
        """
        Inicializa el clasificador
        
        Args:
            config: Diccionario con configuración (si None, autodetecta)
        """
        
        # Usar config auto-detectada o la proporcionada
        self.config = config if config else detectar_hardware()
        
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.model_name = self.config['model_name']
        self.confidence_threshold = -35.0
        self.use_ivf = self.config['use_ivf']
        self.clean_frequency = self.config['clean_frequency']
        
        logger.info("="*80)
        logger.info("📦 CARGANDO MODELO DE EMBEDDINGS")
        logger.info("="*80 + "\n")
        
        # Cargar modelo
        try:
            logger.info(f"Cargando: sentence-transformers/{self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Modelo cargado exitosamente")
            logger.info(f"   Dimensión embedding: {self.embedding_dim}")
            logger.info(f"   Device: {self.device.upper()}\n")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
        
        # Estructuras de datos
        self.categorias = []
        self.faiss_index = None
        self.categoria_mapping = []
    
    def cargar_taxonomia(self, ruta_json: str) -> bool:
        """
        Carga y prepara la taxonomía CGIC
        
        Args:
            ruta_json: Ruta al archivo taxonomia_optimizada.json
            
        Returns:
            True si carga exitosa, False si error
        """
        
        logger.info("="*80)
        logger.info("📂 CARGANDO TAXONOMÍA CGIC")
        logger.info("="*80 + "\n")
        
        try:
            logger.info(f"Leyendo: {ruta_json}")
            with open(ruta_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.categorias = data.get('categorias', [])
            logger.info(f"✅ Taxonomía cargada: {len(self.categorias)} categorías\n")
            
            # Construir índice FAISS
            self._construir_indice_faiss()
            return True
            
        except FileNotFoundError:
            logger.error(f"❌ Archivo no encontrado: {ruta_json}")
            logger.error("   Ejecuta primero: python preparar_taxonomia.py")
            return False
        except json.JSONDecodeError:
            logger.error(f"❌ Error en formato JSON de {ruta_json}")
            return False
        except Exception as e:
            logger.error(f"❌ Error inesperado: {e}")
            return False
    
    def _construir_indice_faiss(self) -> None:
        """
        Construye índice FAISS con embeddings de categorías
        Usa IVFFlat si disponible para búsquedas más rápidas
        """
        
        logger.info("="*80)
        logger.info("⚡ CONSTRUYENDO ÍNDICE FAISS")
        logger.info("="*80 + "\n")
        
        embeddings_lista = []
        self.categoria_mapping = []
        
        # Extraer textos de categorías
        textos = [cat['texto_para_embedding'] for cat in self.categorias]
        logger.info(f"Generando {len(textos)} embeddings en batches...")
        logger.info(f"Batch size: {self.batch_size}\n")
        
        # Generar embeddings en batches grandes
        for i in tqdm(
            range(0, len(textos), self.batch_size),
            desc="Embeddeando categorías",
            unit="batch"
        ):
            batch_textos = textos[i:i+self.batch_size]
            
            embeddings = self.model.encode(
                batch_textos,
                convert_to_tensor=False,
                batch_size=self.batch_size,
                show_progress_bar=False
            ).astype(np.float32)
            
            embeddings_lista.extend(embeddings)
        
        # Crear mapping
        for categoria in self.categorias:
            self.categoria_mapping.append({
                'id': categoria['id'],
                'sector': categoria['sector'],
                'categoria': categoria['categoria'],
                'descripcion': categoria.get('descripcion', ''),
                'keywords': categoria.get('keywords', [])
            })
        
        # Construir índice FAISS
        embeddings_array = np.array(embeddings_lista)
        logger.info(f"\nConstructs índice FAISS...")
        
        if self.use_ivf and len(embeddings_array) > 100:
            # Usar IndexIVFFlat para búsquedas más rápidas
            nlist = max(50, len(embeddings_array) // 50)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                nlist
            )
            self.faiss_index.train(embeddings_array)
            self.faiss_index.add(embeddings_array)
            logger.info(f"✅ IndexIVFFlat creado ({nlist} listas)")
        else:
            # Usar IndexFlatL2 (más simple)
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index.add(embeddings_array)
            logger.info(f"✅ IndexFlatL2 creado")
        
        logger.info(f"   Total de vectores: {self.faiss_index.ntotal}")
        logger.info(f"   Dimensión: {self.embedding_dim}\n")
    
    def procesar_csv(
        self,
        ruta_csv: str,
        ruta_salida: str,
        columna_titular: str = 'headline',
        muestra_max: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Procesa CSV de noticias y genera clasificaciones
        
        Args:
            ruta_csv: Ruta al CSV con noticias
            ruta_salida: Ruta para guardar resultados
            columna_titular: Nombre de columna con titulares
            muestra_max: Procesar solo N filas (None = todas)
            
        Returns:
            Tupla (total_procesadas, total_clasificadas)
        """
        
        logger.info("="*80)
        logger.info("📰 PROCESANDO NOTICIAS")
        logger.info("="*80 + "\n")
        
        # Cargar CSV
        logger.info(f"Cargando CSV: {ruta_csv}")
        try:
            if muestra_max:
                df = pd.read_csv(ruta_csv, nrows=muestra_max, low_memory=False)
                logger.info(f"✅ CSV cargado (muestra: {muestra_max} filas)\n")
            else:
                df = pd.read_csv(ruta_csv, low_memory=False)
                logger.info(f"✅ CSV cargado ({len(df):,} filas)\n")
        except FileNotFoundError:
            logger.error(f"❌ No encontrado: {ruta_csv}")
            return 0, 0
        except Exception as e:
            logger.error(f"❌ Error cargando CSV: {e}")
            return 0, 0
        
        total_registros = len(df)
        
        # Validar columna de titulares
        if columna_titular not in df.columns:
            logger.error(f"❌ Columna no encontrada: {columna_titular}")
            logger.error(f"   Columnas disponibles: {list(df.columns)}")
            return 0, 0
        
        logger.info(f"Procesando con configuración:")
        logger.info(f"  • Batch Size: {self.batch_size}")
        logger.info(f"  • Device: {self.device.upper()}")
        logger.info(f"  • Modelo: {self.model_name}")
        logger.info(f"  • Threshold: {self.confidence_threshold}\n")
        
        resultados = []
        clasificados = 0
        inicio = time.time()
        similitudes = []
        
        # Procesamiento principal
        for batch_start in tqdm(
            range(0, total_registros, self.batch_size),
            desc="Procesando noticias",
            unit="batch",
            total=(total_registros + self.batch_size - 1) // self.batch_size
        ):
            batch_end = min(batch_start + self.batch_size, total_registros)
            batch = df.iloc[batch_start:batch_end]
            
            # Obtener titulares
            titulares = batch[columna_titular].fillna('').tolist()
            
            try:
                # Generar embeddings
                embeddings_batch = self.model.encode(
                    titulares,
                    convert_to_tensor=False,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                ).astype(np.float32)
                
                # Búsqueda en FAISS
                distances, indices = self.faiss_index.search(
                    embeddings_batch,
                    k=1
                )
                
                # Procesar resultados
                for idx_batch, (_, row) in enumerate(batch.iterrows()):
                    dist = distances[idx_batch][0]
                    idx_categoria = indices[idx_batch][0]
                    
                    if idx_categoria != -1:
                        # Convertir distancia L2 a similitud
                        similitud = 1 - (dist ** 2) / 2
                        similitudes.append(similitud)
                        
                        # Mostrar primeras 5 muestras
                        if batch_start == 0 and idx_batch < 5:
                            mapping = self.categoria_mapping[idx_categoria]
                            logger.info(f"MUESTRA {idx_batch + 1}:")
                            logger.info(f"  Titular: '{titulares[idx_batch][:70]}...'")
                            logger.info(f"  Sector: {mapping['sector']}")
                            logger.info(f"  Categoría: {mapping['categoria']}")
                            logger.info(f"  Score: {similitud:.4f}\n")
                        
                        # Guardar si supera threshold
                        if similitud >= self.confidence_threshold:
                            mapping = self.categoria_mapping[idx_categoria]
                            resultado_row = {
                                'date': row.get('date', ''),
                                'headline': row.get('headline', ''),
                                'outlet': row.get('outlet', ''),
                                'sector': mapping['sector'],
                                'categoria': mapping['categoria'],
                                'categoria_id': mapping['id'],
                                'confidence_score': float(similitud)
                            }
                            resultados.append(resultado_row)
                            clasificados += 1
                
                del embeddings_batch
                
            except Exception as e:
                logger.error(f"Error en batch {batch_start}-{batch_end}: {e}")
                continue
            
            # Limpiar memoria según frecuencia configurada
            if batch_start % (self.batch_size * self.clean_frequency) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Guardar resultados
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(ruta_salida, index=False, encoding='utf-8')
        
        # Estadísticas finales
        tiempo_total = time.time() - inicio
        velocidad = total_registros / tiempo_total if tiempo_total > 0 else 0
        tasa = (clasificados / total_registros * 100) if total_registros > 0 else 0
        
        logger.info("\n" + "="*80)
        logger.info("✅ PROCESAMIENTO FINALIZADO")
        logger.info("="*80 + "\n")
        
        logger.info("📊 RESULTADOS:")
        logger.info(f"  Tiempo total: {tiempo_total:.2f}s ({tiempo_total/60:.2f} min)")
        logger.info(f"  Velocidad: {velocidad:,.0f} noticias/segundo")
        logger.info(f"  Total registros: {total_registros:,}")
        logger.info(f"  Clasificadas: {clasificados:,} ({tasa:.1f}%)")
        logger.info(f"  Archivo resultado: {ruta_salida}\n")
        
        if similitudes:
            logger.info("📈 ESTADÍSTICAS DE SIMILITUD:")
            logger.info(f"  Mínima: {min(similitudes):.4f}")
            logger.info(f"  Máxima: {max(similitudes):.4f}")
            logger.info(f"  Promedio: {np.mean(similitudes):.4f}")
            logger.info(f"  Mediana: {np.median(similitudes):.4f}")
            logger.info(f"  Desv. Est: {np.std(similitudes):.4f}")
            logger.info(f"  Threshold: {self.confidence_threshold:.4f}\n")
        
        logger.info("="*80 + "\n")
        
        return total_registros, clasificados


# ==================== FUNCIÓN PRINCIPAL ====================
def main():
    """
    Script principal
    Ejecuta el flujo completo de clasificación
    """
    
    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*78 + "║")
    logger.info("║" + "  CLASIFICADOR DE NOTICIAS CGIC - VERSIÓN FINAL OPTIMIZADA".center(78) + "║")
    logger.info("║" + " "*78 + "║")
    logger.info("╚" + "="*78 + "╝")
    
    # Rutas
    RUTA_CSV = "noticias_filtradas.csv"
    RUTA_TAXONOMIA = "taxonomia_optimizada.json"
    RUTA_SALIDA = "noticias_clasificadas_produccion.csv"
    
    # Validar archivos
    logger.info("\n" + "="*80)
    logger.info("🔐 VALIDANDO ARCHIVOS DE ENTRADA")
    logger.info("="*80 + "\n")
    
    if not Path(RUTA_CSV).exists():
        logger.error(f"❌ No encontrado: {RUTA_CSV}")
        logger.error(f"   Ruta esperada: {Path(RUTA_CSV).absolute()}")
        return False
    logger.info(f"✅ Encontrado: {RUTA_CSV}")
    
    if not Path(RUTA_TAXONOMIA).exists():
        logger.error(f"❌ No encontrado: {RUTA_TAXONOMIA}")
        logger.error("   Ejecuta primero: python preparar_taxonomia.py")
        return False
    logger.info(f"✅ Encontrado: {RUTA_TAXONOMIA}\n")
    
    # Crear clasificador (con autodetección)
    try:
        config = detectar_hardware()
        clasificador = ClasificadorCGICFinal(config)
    except Exception as e:
        logger.error(f"❌ Error inicializando clasificador: {e}")
        return False
    
    # Cargar taxonomía
    if not clasificador.cargar_taxonomia(RUTA_TAXONOMIA):
        return False
    
    # Procesar noticias
    total, clasificados = clasificador.procesar_csv(
        RUTA_CSV,
        RUTA_SALIDA,
        columna_titular='headline',
        muestra_max=None  # Cambiar a None para procesar TODO
    )
    
    if clasificados > 0:
        logger.info("✅ ¡CLASIFICACIÓN COMPLETADA EXITOSAMENTE!")
        logger.info(f"\n📥 Próximos pasos:")
        logger.info(f"   1. Revisar resultados: {RUTA_SALIDA}")
        logger.info(f"   2. Cambiar muestra_max=None en el script para procesar 6M noticias")
        logger.info(f"   3. Ejecutar nuevamente para producción\n")
        return True
    else:
        logger.warning("⚠️  No se clasificó ninguna noticia")
        logger.warning("    Posible causa: threshold muy alto")
        logger.warning("    Reduce confidence_threshold a 0.20\n")
        return False


# ==================== PUNTO DE ENTRADA ====================
if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("\n\n❌ Proceso cancelado por usuario")
        exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error no controlado: {e}")
        import traceback
        traceback.print_exc()
        exit(1)