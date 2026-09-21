"""TrustVoice AI P3 registered-speaker embedding and FAISS search."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import torch
except ImportError:
    torch = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        EncoderClassifier = None

BASE_DIR = Path(__file__).resolve().parent
REGISTRY_DIR = BASE_DIR / "models" / "speaker_registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
INDEX_FILE = REGISTRY_DIR / "speakers.faiss"

SAMPLE_RATE = 16000
MATCH_THRESHOLD = 0.65
EMBEDDING_DIM = 192
_MODEL = None


def _ensure_registry() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        REGISTRY_FILE.write_text(
            json.dumps({"version": 1, "speakers": {}}, indent=2),
            encoding="utf-8",
        )


def load_registry() -> dict[str, Any]:
    _ensure_registry()
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "speakers": {}}


def save_registry(registry: dict[str, Any]) -> None:
    _ensure_registry()
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _require_models() -> None:
    if EncoderClassifier is None:
        raise RuntimeError("SpeechBrain is not installed. Run: pip install speechbrain")
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    if librosa is None:
        raise RuntimeError("librosa is not installed.")


def _get_model():
    global _MODEL
    _require_models()
    if _MODEL is None:
        _MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(BASE_DIR / "models" / "ecapa_voxceleb"),
            run_opts={"device": "cpu"},
        )
    return _MODEL


def decode_audio(raw: bytes, filename: str) -> np.ndarray:
    _require_models()
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    if suffix not in allowed:
        raise ValueError("Supported speaker audio: WAV, MP3, M4A, FLAC, OGG, WEBM.")
    if not raw:
        raise ValueError("Empty audio input.")

    y, _ = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y)

    if y.size < SAMPLE_RATE:
        raise ValueError("Audio must contain at least one second of speech.")

    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals):
        y = np.concatenate([y[a:b] for a, b in intervals])

    if y.size < SAMPLE_RATE:
        raise ValueError("Not enough speech after silence removal.")

    max_samples = SAMPLE_RATE * 20
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start:start + max_samples]

    return np.ascontiguousarray(y, dtype=np.float32)


def create_embedding(raw: bytes, filename: str) -> np.ndarray:
    audio = decode_audio(raw, filename)
    model = _get_model()
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        emb = model.encode_batch(waveform)

    emb = emb.detach().cpu().numpy().reshape(-1).astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm <= 0:
        raise RuntimeError("Speaker embedding has zero norm.")
    return (emb / norm).astype(np.float32)


def _require_faiss() -> None:
    if faiss is None:
        raise RuntimeError("FAISS is not installed. Run: pip install faiss-cpu")


def _load_index():
    _require_faiss()
    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        if index.d != EMBEDDING_DIM:
            raise RuntimeError("Existing FAISS index has incompatible embedding dimension.")
        return index
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def rebuild_index() -> int:
    _require_faiss()
    registry = load_registry()
    speakers = list(registry.get("speakers", {}).values())
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    if speakers:
        matrix = np.asarray([s["embedding"] for s in speakers], dtype=np.float32)
        faiss.normalize_L2(matrix)
        index.add(matrix)

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    return int(index.ntotal)


def enroll_speaker(name: str, samples: list[tuple[bytes, str]]) -> dict[str, Any]:
    _require_faiss()
    name = name.strip()
    if not name:
        raise ValueError("Speaker name is required.")
    if not samples:
        raise ValueError("At least one voice sample is required.")

    embeddings = []
    hashes = []

    for raw, filename in samples:
        embeddings.append(create_embedding(raw, filename))
        hashes.append(hashlib.sha256(raw).hexdigest()[:16])

    mean = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 0:
        raise RuntimeError("Could not construct a stable speaker profile.")
    mean = (mean / norm).astype(np.float32)

    registry = load_registry()
    speakers = registry.setdefault("speakers", {})
    speaker_id = hashlib.sha256(name.lower().encode("utf-8")).hexdigest()[:16]

    speakers[speaker_id] = {
        "id": speaker_id,
        "name": name,
        "embedding": mean.tolist(),
        "samples": len(samples),
        "sample_hashes": hashes,
    }

    save_registry(registry)
    rebuild_index()

    return {
        "speaker_id": speaker_id,
        "name": name,
        "samples": len(samples),
        "embedding_dimension": int(mean.shape[0]),
        "index_size": len(speakers),
    }


def match_speaker(raw: bytes, filename: str) -> dict[str, Any]:
    _require_faiss()
    registry = load_registry()
    speakers = list(registry.get("speakers", {}).values())

    if not speakers:
        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "similarity": 0.0,
            "reason": "No registered speakers.",
        }

    index = _load_index()
    if index.ntotal != len(speakers):
        rebuild_index()
        index = _load_index()

    query = create_embedding(raw, filename).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)

    similarities, indices = index.search(query, min(3, index.ntotal))
    best = float(similarities[0][0])
    idx = int(indices[0][0])

    if idx < 0:
        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "similarity": 0.0,
            "reason": "No vector match.",
        }

    speaker = speakers[idx]

    return {
        "available": True,
        "matched": best >= MATCH_THRESHOLD,
        "speaker": speaker["name"],
        "speaker_id": speaker["id"],
        "similarity": round(float(np.clip(best, -1.0, 1.0)) * 100.0, 2),
        "cosine_similarity": round(best, 5),
        "threshold": MATCH_THRESHOLD,
        "method": "ECAPA-TDNN speaker embedding + FAISS inner-product search",
        "registered_speakers": len(speakers),
        "top_matches": [
            {
                "speaker": speakers[int(i)]["name"],
                "similarity": round(float(score) * 100.0, 2),
            }
            for score, i in zip(similarities[0], indices[0])
            if int(i) >= 0
        ],
    }


def list_registered_speakers() -> list[dict[str, Any]]:
    registry = load_registry()
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "samples": s.get("samples", 0),
        }
        for s in registry.get("speakers", {}).values()
    ]


def delete_speaker(speaker_id: str) -> bool:
    registry = load_registry()
    speakers = registry.get("speakers", {})
    if speaker_id not in speakers:
        return False

    del speakers[speaker_id]
    save_registry(registry)
    rebuild_index()
    return True


def compare_speaker_audio(
    reference_raw: bytes,
    reference_name: str,
    target_raw: bytes,
    target_name: str,
) -> dict[str, Any]:
    ref = create_embedding(reference_raw, reference_name)
    target = create_embedding(target_raw, target_name)
    cosine = float(np.clip(np.dot(ref, target), -1.0, 1.0))

    return {
        "available": True,
        "score": round(cosine * 100.0, 2),
        "cosine": round(cosine, 5),
        "reference_file": reference_name,
        "target_file": target_name,
        "method": "ECAPA-TDNN speaker embeddings + cosine similarity",
        "interpretation": "Similarity signal only; not proof of speaker identity or authenticity.",
    }
