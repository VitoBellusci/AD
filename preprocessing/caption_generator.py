from typing import Dict, Any

class CaptionGenerator:
    """
    Generatore di didascalie aggiornato per i metadati numerici del Google Cartoon Set.
    Le categorie visive (che nel CSV sono numeri) vengono trattate come token testuali unici.
    """
    def __init__(self, template: str = None):
        # Template deterministico con 5 attributi fortemente visibili
        self.template = template or (
            "avatar with face {face_color}, hair {hair}, "
            "eyes {eye_color}, glasses {glasses}, and facial hair {facial_hair}"
        )

    def generate(self, metadata: Dict[str, Any]) -> str:
        try:
            # .format() popola la stringa con i valori testuali dei numeri (es. '4', '98')
            return self.template.format(
                face_color=metadata.get('face_color', '0'),
                hair=metadata.get('hair', '0'),
                eye_color=metadata.get('eye_color', '0'),
                glasses=metadata.get('glasses', '0'),
                facial_hair=metadata.get('facial_hair', '0')
            )
        except KeyError:
            # Fallback di sicurezza in caso di dizionario anomalo
            return "avatar with default attributes"

    def __repr__(self):
        return f"CaptionGenerator(template='{self.template}')"