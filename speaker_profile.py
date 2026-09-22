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
            json.dumps(
                {"version": 1, "speakers": {}},
                indent=2,
            ),
            encoding="utf-8",
        )


def load_registry() -> dict[str, Any]:
    _ensure_registry()

    try:
        data = json.loads(
            REGISTRY_FILE.read_text(encoding="utf-8")
        )

        if not isinstance(data, dict):
            raise ValueError("Registry root must be an object.")

        data.setdefault("version", 1)
        data.setdefault("speakers", {})

        if not isinstance(data["speakers"], dict):
            data["speakers"] = {}

        return data

    except Exception:
        return {
            "version": 1,
            "speakers": {},
        }


def save_registry(registry: dict[str, Any]) -> None:
    _ensure_registry()

    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


def _require_models() -> None:
    if EncoderClassifier is None:
        raise RuntimeError(
            "SpeechBrain is not installed. "
            "Run: pip install speechbrain"
        )

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
            savedir=str(
                BASE_DIR / "models" / "ecapa_voxceleb"
            ),
            run_opts={"device": "cpu"},
        )

    return _MODEL


def decode_audio(
    raw: bytes,
    filename: str,
) -> np.ndarray:
    """Decode common audio/video containers to mono 16 kHz float32 speech."""
    _require_models()
    if not raw:
        raise ValueError("Empty audio input.")

    suffix = Path(filename).suffix.lower() or ".bin"
    tmpdir = None
    try:
        import imageio_ffmpeg
        import subprocess
        import tempfile
        tmpdir = tempfile.TemporaryDirectory()
        work = Path(tmpdir.name)
        source = work / f"input{suffix}"
        wav = work / "speaker_16k.wav"
        source.write_bytes(raw)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
            detail = proc.stderr.strip()[-700:] or "Unsupported or corrupt audio."
            raise ValueError(f"Could not decode audio: {detail}")
        audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Could not decode the audio. Please upload a valid speech recording.") from exc
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if audio.size < SAMPLE_RATE:
        raise ValueError("Audio must contain at least one second of speech.")
    try:
        intervals = librosa.effects.split(audio, top_db=30)
    except Exception:
        intervals = []
    if len(intervals):
        audio = np.concatenate([audio[start:end] for start, end in intervals])
    if audio.size < SAMPLE_RATE:
        raise ValueError("Not enough speech after silence removal.")
    max_samples = SAMPLE_RATE * 30
    if len(audio) > max_samples:
        start = (len(audio) - max_samples) // 2
        audio = audio[start:start + max_samples]
    return np.ascontiguousarray(audio, dtype=np.float32)


def create_embedding(
    raw: bytes,
    filename: str,
) -> np.ndarray:
    """Create a normalized ECAPA speaker embedding."""

    audio = decode_audio(
        raw,
        filename,
    )

    model = _get_model()

    waveform = torch.tensor(
        audio,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_batch(
            waveform
        )

    embedding = (
        embedding
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float32)
    )

    if embedding.size != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected speaker embedding dimension: "
            f"{embedding.size}. "
            f"Expected {EMBEDDING_DIM}."
        )

    norm = np.linalg.norm(embedding)

    if norm <= 0:
        raise RuntimeError(
            "Speaker embedding has zero norm."
        )

    return (
        embedding / norm
    ).astype(np.float32)


def _require_faiss() -> None:
    if faiss is None:
        raise RuntimeError(
            "FAISS is not installed. "
            "Run: pip install faiss-cpu"
        )


def _load_index():
    """Load FAISS index or create an empty one."""

    _require_faiss()

    if INDEX_FILE.exists():
        try:
            index = faiss.read_index(
                str(INDEX_FILE)
            )

            if index.d != EMBEDDING_DIM:
                raise RuntimeError(
                    "Existing FAISS index has "
                    "incompatible embedding dimension."
                )

            return index

        except Exception as exc:
            raise RuntimeError(
                "Could not load the speaker FAISS index. "
                "Delete models/speaker_registry/"
                "speakers.faiss and rebuild the registry."
            ) from exc

    return faiss.IndexFlatIP(
        EMBEDDING_DIM
    )


def rebuild_index() -> int:
    """Rebuild FAISS index from registered speakers."""

    _require_faiss()

    registry = load_registry()

    speakers = list(
        registry
        .get("speakers", {})
        .values()
    )

    index = faiss.IndexFlatIP(
        EMBEDDING_DIM
    )

    valid_embeddings = []

    for speaker in speakers:
        embedding = speaker.get(
            "embedding"
        )

        if not isinstance(
            embedding,
            list,
        ):
            continue

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size != EMBEDDING_DIM:
            continue

        norm = np.linalg.norm(vector)

        if norm <= 0:
            continue

        vector = vector / norm

        valid_embeddings.append(
            vector
        )

    if valid_embeddings:
        matrix = np.asarray(
            valid_embeddings,
            dtype=np.float32,
        )

        faiss.normalize_L2(matrix)

        index.add(matrix)

    REGISTRY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    return int(index.ntotal)


def enroll_speaker(
    name: str,
    samples: list[tuple[bytes, str]],
) -> dict[str, Any]:
    """Create or replace a registered speaker profile."""

    _require_faiss()

    name = name.strip()

    if not name:
        raise ValueError(
            "Speaker name is required."
        )

    if not samples:
        raise ValueError(
            "At least one voice sample is required."
        )

    embeddings: list[np.ndarray] = []
    hashes: list[str] = []

    for raw, filename in samples:
        embeddings.append(
            create_embedding(
                raw,
                filename,
            )
        )

        hashes.append(
            hashlib.sha256(
                raw
            ).hexdigest()[:16]
        )

    mean_embedding = np.mean(
        np.stack(embeddings),
        axis=0,
    )

    norm = np.linalg.norm(
        mean_embedding
    )

    if norm <= 0:
        raise RuntimeError(
            "Could not construct a stable "
            "speaker profile."
        )

    mean_embedding = (
        mean_embedding / norm
    ).astype(np.float32)

    registry = load_registry()

    speakers = registry.setdefault(
        "speakers",
        {},
    )

    speaker_id = hashlib.sha256(
        name.lower().encode("utf-8")
    ).hexdigest()[:16]

    speakers[speaker_id] = {
        "id": speaker_id,
        "name": name,
        "embedding": mean_embedding.tolist(),
        "samples": len(samples),
        "sample_hashes": hashes,
    }

    save_registry(registry)

    rebuild_index()

    return {
        "speaker_id": speaker_id,
        "name": name,
        "samples": len(samples),
        "embedding_dimension": int(
            mean_embedding.shape[0]
        ),
        "index_size": len(speakers),
    }


def match_speaker(
    raw: bytes,
    filename: str,
) -> dict[str, Any]:
    """Find closest registered speaker."""

    _require_faiss()

    registry = load_registry()

    speakers = list(
        registry
        .get("speakers", {})
        .values()
    )

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

    if index.ntotal == 0:
        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "similarity": 0.0,
            "reason": (
                "Speaker index contains "
                "no valid embeddings."
            ),
        }

    query = create_embedding(
        raw,
        filename,
    )

    query = query.reshape(
        1,
        -1,
    ).astype(np.float32)

    faiss.normalize_L2(query)

    similarities, indices = index.search(
        query,
        min(3, index.ntotal),
    )

    best = float(
        similarities[0][0]
    )

    idx = int(
        indices[0][0]
    )

    if idx < 0 or idx >= len(speakers):
        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "similarity": 0.0,
            "reason": "No valid vector match.",
        }

    speaker = speakers[idx]

    return {
        "available": True,
        "matched": (
            best >= MATCH_THRESHOLD
        ),
        "speaker": speaker["name"],
        "speaker_id": speaker["id"],
        "similarity": round(
            float(
                np.clip(
                    best,
                    -1.0,
                    1.0,
                )
            ) * 100.0,
            2,
        ),
        "cosine_similarity": round(
            best,
            5,
        ),
        "threshold": MATCH_THRESHOLD,
        "method": (
            "ECAPA-TDNN speaker embedding + "
            "FAISS inner-product search"
        ),
        "registered_speakers": len(
            speakers
        ),
        "top_matches": [
            {
                "speaker": speakers[
                    int(index)
                ]["name"],
                "similarity": round(
                    float(score) * 100.0,
                    2,
                ),
            }
            for score, index in zip(
                similarities[0],
                indices[0],
            )
            if 0 <= int(index) < len(speakers)
        ],
        "interpretation": (
            "Speaker similarity is a matching "
            "signal only. It is not proof of "
            "identity or voice authenticity."
        ),
    }


def list_registered_speakers() -> list[
    dict[str, Any]
]:
    registry = load_registry()

    return [
        {
            "id": speaker["id"],
            "name": speaker["name"],
            "samples": speaker.get(
                "samples",
                0,
            ),
        }
        for speaker in registry
        .get("speakers", {})
        .values()
    ]


def delete_speaker(
    speaker_id: str,
) -> bool:
    registry = load_registry()

    speakers = registry.get(
        "speakers",
        {},
    )

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
    """Compare two audio files using ECAPA embeddings."""

    reference_embedding = create_embedding(
        reference_raw,
        reference_name,
    )

    target_embedding = create_embedding(
        target_raw,
        target_name,
    )

    cosine = float(
        np.clip(
            np.dot(
                reference_embedding,
                target_embedding,
            ),
            -1.0,
            1.0,
        )
    )

    return {
        "available": True,
        "score": round(
            cosine * 100.0,
            2,
        ),
        "cosine": round(
            cosine,
            5,
        ),
        "reference_file": reference_name,
        "target_file": target_name,
        "method": (
            "ECAPA-TDNN speaker embeddings + "
            "cosine similarity"
        ),
        "interpretation": (
            "Similarity signal only; not proof "
            "of speaker identity or authenticity."
        ),
    }