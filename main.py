import src.acquisition as acquisition
import src.preprocessing as preprocessing
import src.rag as rag


def main():
    """
    Función principal que orquesta el flujo de trabajo del proyecto.
    """

    print("1. Iniciando adquisición...")
    datos = acquisition.obtener_datos()

    print("2. Iniciando preprocesamiento...")
    datos_limpios = preprocessing.limpiar_datos(datos)

    print("3. Ejecutando RAG...")
    rag.consultar(datos_limpios)

    print("¡Proceso terminado con éxito!")


if __name__ == "__main__":
    main()
