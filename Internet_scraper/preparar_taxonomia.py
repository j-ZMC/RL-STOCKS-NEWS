"""
Prepara la taxonomía CGIC para clasificación de embeddings
Convierte formato JSON complejo a estructura optimizada para búsqueda vectorial
"""

import json
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# cgic_taxonomy
class PreparadorTaxonomia:
    """Convierte JSON de taxonomía a formato optimizado para embeddings"""
    
    def __init__(self, ruta_json: str):
        self.ruta_json = ruta_json
        self.taxonomia_bruta = {}
        self.categorias_optimizadas = []
        
    def cargar_json(self) -> bool:
        """Carga el JSON de taxonomía"""
        try:
            with open(self.ruta_json, 'r', encoding='utf-8') as f:
                self.taxonomia_bruta = json.load(f)
            logger.info(f"✅ Taxonomía cargada: {len(self.taxonomia_bruta)} sectores")
            return True
        except Exception as e:
            logger.error(f"❌ Error cargando JSON: {e}")
            return False
    
    def procesar_taxonomia(self) -> List[Dict]:
        """
        Procesa taxonomía compleja en lista plana optimizada
        Cada elemento = 1 categoría con contexto completo para embedding
        """
        logger.info("Procesando taxonomía...")
        categorias = []
        
        for sector_nombre, sector_data in self.taxonomia_bruta.items():
            # Saltar notas de sector
            if sector_nombre.startswith('_'):
                continue
            
            for categoria_nombre, categoria_data in sector_data.items():
                # Saltar notas de categoría
                if categoria_nombre.startswith('_'):
                    continue
                
                # Validar que sea un dict (no string de metadata)
                if not isinstance(categoria_data, dict):
                    continue
                
                # Extraer información
                descripcion = categoria_data.get('description', '')
                keywords = categoria_data.get('keywords', [])
                evitar_confusion = categoria_data.get('avoid_confusion_with', [])
                
                # Crear texto de búsqueda enriquecido
                # Esto es lo que se embebecerá
                texto_busqueda = self._construir_texto_busqueda(
                    sector_nombre,
                    categoria_nombre,
                    descripcion,
                    keywords
                )
                
                # Crear ID único
                categoria_id = f"{sector_nombre.lower().replace(' ', '_')}:{categoria_nombre.lower().replace(' ', '_')}"
                
                elemento = {
                    'id': categoria_id,
                    'sector': sector_nombre,
                    'categoria': categoria_nombre,
                    'descripcion': descripcion,
                    'keywords': keywords,
                    'keywords_count': len(keywords),
                    'texto_para_embedding': texto_busqueda,
                    'evitar_confusion': evitar_confusion,
                    'nivel': 'CATEGORIA'  # Para posibles sub-niveles futuros
                }
                
                categorias.append(elemento)
        
        self.categorias_optimizadas = categorias
        logger.info(f"✅ {len(categorias)} categorías procesadas")
        return categorias
    
    def _construir_texto_busqueda(
        self,
        sector: str,
        categoria: str,
        descripcion: str,
        keywords: List[str]
    ) -> str:
        """
        Construye texto optimizado para embeddings
        Pesa más los términos específicos (keywords) que la descripción genérica
        """
        # Formato: [SECTOR] Categoría - Descripción + Keywords explícitos
        # Los keywords se repiten para dar peso en el embedding
        
        keywords_str = " ".join(keywords)
        
        # Repetir keywords 2x para dar más peso en el embedding
        texto = f"""
[{sector}]
{categoria}

{descripcion}

Términos clave: {keywords_str}

Relevantes: {keywords_str}
"""
        return texto.strip()
    
    def exportar_csv(self, ruta_salida: str) -> bool:
        """Exporta categorías a CSV para análisis"""
        try:
            df = pd.DataFrame(self.categorias_optimizadas)
            df.to_csv(ruta_salida, index=False, encoding='utf-8')
            logger.info(f"✅ Exportado a CSV: {ruta_salida}")
            return True
        except Exception as e:
            logger.error(f"❌ Error exportando CSV: {e}")
            return False
    
    def exportar_json_optimizado(self, ruta_salida: str) -> bool:
        """Exporta en formato JSON optimizado para clasificador"""
        try:
            estructura = {
                'version': '2.0',
                'fecha_generacion': pd.Timestamp.now().isoformat(),
                'total_categorias': len(self.categorias_optimizadas),
                'categorias': self.categorias_optimizadas
            }
            
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                json.dump(estructura, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Exportado JSON optimizado: {ruta_salida}")
            return True
        except Exception as e:
            logger.error(f"❌ Error exportando JSON: {e}")
            return False
    
    def generar_reporte(self) -> Dict:
        """Genera reporte de análisis de taxonomía"""
        df = pd.DataFrame(self.categorias_optimizadas)
        
        reporte = {
            'total_categorias': len(self.categorias_optimizadas),
            'total_sectores': df['sector'].nunique(),
            'sectores': df['sector'].unique().tolist(),
            'categorias_por_sector': df.groupby('sector').size().to_dict(),
            'total_keywords': df['keywords_count'].sum(),
            'promedio_keywords_por_categoria': df['keywords_count'].mean(),
            'min_keywords': df['keywords_count'].min(),
            'max_keywords': df['keywords_count'].max()
        }
        
        return reporte
    
    def imprimir_reporte(self):
        """Imprime reporte formateado"""
        reporte = self.generar_reporte()
        
        logger.info("\n" + "="*60)
        logger.info("📊 REPORTE DE TAXONOMÍA")
        logger.info("="*60)
        logger.info(f"\n✅ Análisis Completado:")
        logger.info(f"  Total de categorías: {reporte['total_categorias']}")
        logger.info(f"  Total de sectores: {reporte['total_sectores']}")
        logger.info(f"  Total de keywords: {reporte['total_keywords']}")
        logger.info(f"  Promedio keywords/categoría: {reporte['promedio_keywords_por_categoria']:.1f}")
        
        logger.info(f"\n📋 Sectores ({reporte['total_sectores']}):")
        for sector, count in sorted(reporte['categorias_por_sector'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  [{count:2d}] {sector}")
        
        logger.info(f"\n{'='*60}\n")
        
        return reporte


def main():
    """Ejecuta preparación de taxonomía"""
    
    logger.info("\n" + "="*60)
    logger.info("PREPARADOR DE TAXONOMÍA CGIC")
    logger.info("="*60 + "\n")
    
    # Archivos
    RUTA_ENTRADA = "GICS_clasificator.json"  # ← Tu archivo descargado
    RUTA_CSV = "taxonomia_optimizada.csv"
    RUTA_JSON = "taxonomia_optimizada.json"
    
    # Procesar
    preparador = PreparadorTaxonomia(RUTA_ENTRADA)
    
    if not preparador.cargar_json():
        logger.error("No se pudo cargar la taxonomía")
        return False
    
    preparador.procesar_taxonomia()
    preparador.exportar_csv(RUTA_CSV)
    preparador.exportar_json_optimizado(RUTA_JSON)
    preparador.imprimir_reporte()
    
    logger.info("✅ Taxonomía lista para clasificación")
    logger.info(f"\nArchivos generados:")
    logger.info(f"  1. {RUTA_CSV} - Para análisis manual")
    logger.info(f"  2. {RUTA_JSON} - Para clasificador (USAR ESTE)")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
