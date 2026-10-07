import re
from typing import Dict, Any, Optional

class _SafeDict(dict):
    """
    Dizionario sicuro che restituisce un fallback descrittivo naturale per 
    eventuali segnaposto imprevisti nel template.
    """
    def __missing__(self, key: str) -> str:
        return "natural"

class CaptionGenerator:
    """
    Generatore deterministico di didascalie in linguaggio naturale per Google Cartoon Set.
    Mappa i metadati numerici delle categorie visive (face_color, hair, eye_color, glasses,
    facial_hair, ecc.) in descrittori semantici naturali, garantendo l'assenza totale di ID numerici.
    """

    FACE_COLORS = {
        "0": "porcelain",
        "1": "brown",
        "2": "light",
        "3": "peach",
        "4": "olive",
        "5": "tan",
        "6": "golden",
        "7": "bronze",
        "8": "chestnut",
        "9": "dark",
        "10": "pale"
    }

    HAIR_STYLES = {
        "0": "short", "1": "straight", "2": "curly", "3": "wavy", "4": "spiky",
        "5": "bob", "6": "bun", "7": "ponytail", "8": "afro", "9": "buzzcut",
        "10": "dreadlocks", "11": "parted", "12": "slicked", "13": "layered", "14": "shaggy",
        "15": "undercut", "16": "mohawk", "17": "pixie", "18": "fringe", "19": "cropped",
        "20": "quiff", "21": "pompadour", "22": "neat", "23": "fade", "24": "braided",
        "25": "messy", "26": "combed", "27": "voluminous", "28": "tapered", "29": "tousled",
        "30": "coily", "31": "textured", "32": "topknot", "33": "cornrows", "34": "braids",
        "35": "wavy bob", "36": "curly bob", "37": "straight bob", "38": "choppy", "39": "feathered",
        "40": "slicked back", "41": "side part", "42": "middle part", "43": "swept", "44": "wild",
        "45": "relaxed", "46": "retro", "47": "cropped wavy", "48": "cropped curly", "49": "short spiky",
        "50": "brushed", "51": "swept back", "52": "curtain", "53": "taper fade", "54": "dreads",
        "55": "twisted", "56": "ringlets", "57": "tight curls", "58": "loose curls", "59": "beach waves",
        "60": "short crop", "61": "blunt cut", "62": "shag", "63": "asymmetrical", "64": "slick",
        "65": "finger waves", "66": "mullet", "67": "soft waves", "68": "pinned", "69": "swept bangs",
        "70": "side braids", "71": "crown braid", "72": "micro braids", "73": "high fade", "74": "low fade",
        "75": "classic part", "76": "casual", "77": "elegant", "78": "swooped", "79": "front flip",
        "80": "spiky crop", "81": "swept quiff", "82": "brushed up", "83": "wispy", "84": "feathered bangs",
        "85": "side curls", "86": "parted bob", "87": "angled bob", "88": "shoulder curls", "89": "wavy shag",
        "90": "curly shag", "91": "textured crop", "92": "buzz", "93": "shaved sides", "94": "clean cut",
        "95": "casual curls", "96": "natural waves", "97": "soft curls", "98": "wavy", "99": "swept waves",
        "100": "styled curls", "101": "high ponytail", "102": "low ponytail", "103": "braided bun", "104": "side bun",
        "105": "curled bob", "106": "tousled waves", "107": "classic waves", "108": "slicked side", "109": "long straight",
        "110": "shoulder length"
    }

    EYE_COLORS = {
        "0": "blue",
        "1": "green",
        "2": "brown",
        "3": "hazel",
        "4": "dark"
    }

    GLASSES_STYLES = {
        "0": "round glasses",
        "1": "square glasses",
        "2": "rectangular glasses",
        "3": "oval glasses",
        "4": "aviator glasses",
        "5": "cat eye glasses",
        "6": "rimmed glasses",
        "7": "rimless glasses",
        "8": "half rim glasses",
        "9": "wire glasses",
        "10": "thick frame glasses",
        "11": "no glasses"
    }

    FACIAL_HAIR_STYLES = {
        "0": "light stubble",
        "1": "mustache",
        "2": "goatee",
        "3": "full beard",
        "4": "short beard",
        "5": "circle beard",
        "6": "chin strap",
        "7": "mutton chops",
        "8": "handlebar mustache",
        "9": "horseshoe mustache",
        "10": "soul patch",
        "11": "thick beard",
        "12": "goatee and mustache",
        "13": "sideburns",
        "14": "no facial hair"
    }

    HAIR_COLORS = {
        "0": "black",
        "1": "dark brown",
        "2": "red",
        "3": "brown",
        "4": "blonde",
        "5": "gray",
        "6": "white",
        "7": "auburn",
        "8": "chestnut",
        "9": "silver"
    }

    FACE_SHAPES = {
        "0": "round",
        "1": "oval",
        "2": "square",
        "3": "heart",
        "4": "narrow",
        "5": "wide",
        "6": "diamond"
    }

    GLASSES_COLORS = {
        "0": "black",
        "1": "brown",
        "2": "blue",
        "3": "green",
        "4": "red",
        "5": "purple",
        "6": "yellow"
    }

    EYEBROW_SHAPES = {
        "0": "arched",
        "1": "curved",
        "2": "straight",
        "3": "rounded",
        "4": "angled",
        "5": "flat",
        "6": "soft arch",
        "7": "high arch",
        "8": "thick arch",
        "9": "thin arch",
        "10": "tapered",
        "11": "steep",
        "12": "s-shaped",
        "13": "natural"
    }

    EYEBROW_THICKNESS = {
        "0": "thin",
        "1": "medium",
        "2": "thick",
        "3": "bold"
    }

    CHIN_LENGTHS = {
        "0": "short",
        "1": "medium",
        "2": "long"
    }

    EYE_ANGLES = {
        "0": "upward",
        "1": "straight",
        "2": "downward"
    }

    EYE_LASHES = {
        "0": "natural eyelashes",
        "1": "prominent eyelashes"
    }

    EYE_LIDS = {
        "0": "single eyelid",
        "1": "double eyelid"
    }

    EYEBROW_WEIGHTS = {
        "0": "light eyebrows",
        "1": "heavy eyebrows"
    }

    EYE_SLANTS = {
        "0": "straight",
        "1": "upturned",
        "2": "downturned"
    }

    EYEBROW_WIDTHS = {
        "0": "narrow",
        "1": "medium",
        "2": "wide"
    }

    EYE_EYEBROW_DISTANCES = {
        "0": "close",
        "1": "average",
        "2": "high"
    }

    DEFAULT_TEMPLATE = (
        "a cartoon avatar with {face_color} skin, {hair} hair, "
        "{eye_color} eyes, {glasses}, and {facial_hair}"
    )

    def __init__(self, template: Optional[str] = None):
        self.template = template or self.DEFAULT_TEMPLATE

    @staticmethod
    def _normalize_metadata(metadata: Any) -> Dict[str, Any]:
        """Normalizza le chiavi del dizionario metadati (lowercase, stripped). Supporta anche Mapping e DataFrame Series."""
        if hasattr(metadata, "to_dict") and callable(metadata.to_dict):
            try:
                metadata = metadata.to_dict()
            except Exception:
                pass
        if hasattr(metadata, "items") and callable(metadata.items):
            try:
                return {str(k).strip().lower(): v for k, v in metadata.items()}
            except Exception:
                pass
        if not isinstance(metadata, dict):
            return {}
        return {str(k).strip().lower(): v for k, v in metadata.items()}

    @classmethod
    def _extract_and_resolve(
        cls,
        normalized_meta: Dict[str, Any],
        aliases: list,
        mapping: Dict[str, str],
        default_key: str,
        fallback_descriptor: str
    ) -> str:
        """
        Estrae un valore dai metadati provando una lista di alias,
        normalizza numeri interi, float, stringhe descrittive pre-esistenti
        e valori fuori range, restituendo tassativamente un descrittore semantico
        senza ID numerici.
        """
        raw_val = None
        for alias in aliases:
            if alias in normalized_meta:
                raw_val = normalized_meta[alias]
                break

        if raw_val is None or str(raw_val).strip() == "":
            raw_val = default_key

        val_str = str(raw_val).strip()
        # Normalizzazione numerica: float interi (es. 98.0 -> "98")
        try:
            f = float(val_str)
            if f.is_integer():
                val_str = str(int(f))
        except (ValueError, OverflowError):
            pass

        # 1. Ricerca diretta per chiave ID (es. "98" -> "wavy")
        if val_str in mapping:
            return mapping[val_str]

        # 2. Se il valore fornito e' gia' un descrittore semantico valido (es. "wavy", "round", "round glasses")
        val_lower = val_str.lower()
        val_clean = val_lower.replace(" hair", "").replace(" style", "").replace(" beard", "").replace(" glasses", "").replace(" eyes", "").replace(" eye", "").strip()
        for v in mapping.values():
            v_lower = v.lower()
            v_clean = v_lower.replace(" hair", "").replace(" style", "").replace(" beard", "").replace(" glasses", "").replace(" eyes", "").replace(" eye", "").strip()
            if (val_lower == v_lower or 
                (val_clean and val_clean == v_lower) or 
                (val_clean and v_clean and val_clean == v_clean) or 
                (v_clean and val_lower == v_clean)):
                return v

        # 3. Fallback semantico per chiavi sconosciute, negative o anomale (nessun ID numerico nel testo)
        return mapping.get(default_key, fallback_descriptor)

    @classmethod
    def get_canonical_prompts(cls) -> list:
        """
        Restituisce una lista di testi canonici che coprono tutti i descrittori semantici
        delle categorie visive e i prompt OOD / di default.
        Garantisce che il vocabolario del tokenizer contenga tutti i token necessari,
        anche se addestrato su un sottoinsieme limitato di dati.
        """
        prompts = [
            cls().generate({}),
            "a blue cartoon avatar with round eyes and exaggerated proportions",
            "a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard",
            "a cartoon avatar with wavy hair and no glasses",
            "a cartoon avatar with natural features and styled look"
        ]
        for mapping in [
            cls.FACE_COLORS, cls.HAIR_STYLES, cls.EYE_COLORS, cls.GLASSES_STYLES,
            cls.FACIAL_HAIR_STYLES, cls.HAIR_COLORS, cls.FACE_SHAPES, cls.GLASSES_COLORS,
            cls.EYEBROW_SHAPES, cls.EYEBROW_THICKNESS, cls.CHIN_LENGTHS, cls.EYE_ANGLES,
            cls.EYE_LASHES, cls.EYE_LIDS, cls.EYEBROW_WEIGHTS, cls.EYE_SLANTS,
            cls.EYEBROW_WIDTHS, cls.EYE_EYEBROW_DISTANCES
        ]:
            prompts.extend(mapping.values())
        return prompts

    def generate(self, metadata: Dict[str, Any]) -> str:
        try:
            norm_meta = self._normalize_metadata(metadata)

            face_desc = self._extract_and_resolve(
                norm_meta, ['face_color', 'face', 'skin', 'face_tone', 'skin_color'],
                self.FACE_COLORS, '0', 'porcelain'
            )
            hair_desc = self._extract_and_resolve(
                norm_meta, ['hair', 'hair_style', 'hairstyle', 'hair_cut'],
                self.HAIR_STYLES, '0', 'short'
            )
            eye_desc = self._extract_and_resolve(
                norm_meta, ['eye_color', 'eyes', 'eye'],
                self.EYE_COLORS, '0', 'blue'
            )
            glasses_desc = self._extract_and_resolve(
                norm_meta, ['glasses', 'glasses_style', 'eyewear'],
                self.GLASSES_STYLES, '11', 'no glasses'
            )
            facial_hair_desc = self._extract_and_resolve(
                norm_meta, ['facial_hair', 'facial_hair_style', 'beard'],
                self.FACIAL_HAIR_STYLES, '14', 'no facial hair'
            )
            hair_color_desc = self._extract_and_resolve(
                norm_meta, ['hair_color', 'hair_shade'],
                self.HAIR_COLORS, '0', 'black'
            )
            face_shape_desc = self._extract_and_resolve(
                norm_meta, ['face_shape', 'shape'],
                self.FACE_SHAPES, '0', 'round'
            )
            glasses_color_desc = self._extract_and_resolve(
                norm_meta, ['glasses_color'],
                self.GLASSES_COLORS, '0', 'black'
            )
            eyebrow_shape_desc = self._extract_and_resolve(
                norm_meta, ['eyebrow_shape'],
                self.EYEBROW_SHAPES, '0', 'arched'
            )
            eyebrow_thickness_desc = self._extract_and_resolve(
                norm_meta, ['eyebrow_thickness'],
                self.EYEBROW_THICKNESS, '1', 'medium'
            )
            chin_length_desc = self._extract_and_resolve(
                norm_meta, ['chin_length'],
                self.CHIN_LENGTHS, '1', 'medium'
            )
            eye_angle_desc = self._extract_and_resolve(
                norm_meta, ['eye_angle'],
                self.EYE_ANGLES, '1', 'straight'
            )
            eye_lashes_desc = self._extract_and_resolve(
                norm_meta, ['eye_lashes', 'eyelashes', 'lashes'],
                self.EYE_LASHES, '0', 'natural eyelashes'
            )
            eye_lid_desc = self._extract_and_resolve(
                norm_meta, ['eye_lid', 'eyelid', 'eyelids'],
                self.EYE_LIDS, '0', 'single eyelid'
            )
            eyebrow_weight_desc = self._extract_and_resolve(
                norm_meta, ['eyebrow_weight', 'eyebrow_tone'],
                self.EYEBROW_WEIGHTS, '0', 'light eyebrows'
            )
            eye_slant_desc = self._extract_and_resolve(
                norm_meta, ['eye_slant', 'slant'],
                self.EYE_SLANTS, '0', 'straight'
            )
            eyebrow_width_desc = self._extract_and_resolve(
                norm_meta, ['eyebrow_width'],
                self.EYEBROW_WIDTHS, '1', 'medium'
            )
            eye_eyebrow_dist_desc = self._extract_and_resolve(
                norm_meta, ['eye_eyebrow_distance', 'eyebrow_distance'],
                self.EYE_EYEBROW_DISTANCES, '1', 'average'
            )

            format_dict = _SafeDict({
                'face_color': face_desc,
                'face': face_desc,
                'skin': face_desc,
                'skin_color': face_desc,
                'face_tone': face_desc,
                'hair': hair_desc,
                'hair_style': hair_desc,
                'hairstyle': hair_desc,
                'hair_cut': hair_desc,
                'eye_color': eye_desc,
                'eyes': eye_desc,
                'eye': eye_desc,
                'glasses': glasses_desc,
                'glasses_style': glasses_desc,
                'eyewear': glasses_desc,
                'facial_hair': facial_hair_desc,
                'facial_hair_style': facial_hair_desc,
                'beard': facial_hair_desc,
                'hair_color': hair_color_desc,
                'hair_shade': hair_color_desc,
                'face_shape': face_shape_desc,
                'shape': face_shape_desc,
                'glasses_color': glasses_color_desc,
                'eyebrow_shape': eyebrow_shape_desc,
                'eyebrow_thickness': eyebrow_thickness_desc,
                'chin_length': chin_length_desc,
                'eye_angle': eye_angle_desc,
                'eye_lashes': eye_lashes_desc,
                'eyelashes': eye_lashes_desc,
                'lashes': eye_lashes_desc,
                'eye_lid': eye_lid_desc,
                'eyelid': eye_lid_desc,
                'eyelids': eye_lid_desc,
                'eyebrow_weight': eyebrow_weight_desc,
                'eye_slant': eye_slant_desc,
                'slant': eye_slant_desc,
                'eyebrow_width': eyebrow_width_desc,
                'eye_eyebrow_distance': eye_eyebrow_dist_desc,
                'eyebrow_distance': eye_eyebrow_dist_desc,
            })

            caption = self.template.format_map(format_dict)
            # Rimozione di sicurezza di eventuali cifre numeriche residue
            caption = re.sub(r'\b\d+\b', '', caption)
            caption = re.sub(r'\s*,\s*', ', ', caption)
            caption = re.sub(r'(,\s*)+', ', ', caption)
            caption = re.sub(r'\s+', ' ', caption).strip(' ,')
            return caption
        except Exception:
            # Fallback di massima sicurezza in caso di eccezioni impreviste
            return "a cartoon avatar with natural features and no glasses"

    def __repr__(self) -> str:
        return f"CaptionGenerator(template='{self.template}')"