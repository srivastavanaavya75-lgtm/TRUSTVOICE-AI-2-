"""
TrustVoice AI
Registered Speaker Profile + Speaker Matching

Speaker matching is an OPTIONAL identity signal.

Design:
- Voice authenticity and speaker identity remain separate.
- Speaker similarity is NOT proof of identity.
- Speaker similarity is NOT proof of voice authenticity.
- SpeechBrain/ECAPA is used when available.
- A lightweight local acoustic fingerprint is used as a fallback
  when SpeechBrain cannot be loaded in the deployment environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REGISTRY_DIR = (
    BASE_DIR
    / "models"
    / "speaker_registry"
)

REGISTRY_FILE = (
    REGISTRY_DIR
    / "registry.json"
)

INDEX_FILE = (
    REGISTRY_DIR
    / "speakers.faiss"
)

SAMPLE_RATE = 16000

MATCH_THRESHOLD = 0.65

EMBEDDING_DIM = 192


# ============================================================
# FEATURE FLAG
# ============================================================
#
# Speaker matching is ENABLED by default.
#
# To explicitly disable it:
#
# TRUSTVOICE_ENABLE_SPEAKER_MATCHING=false
#
# This avoids the old situation where the Registry UI existed
# but enrollment immediately failed because the feature was
# disabled by default.
# ============================================================

SPEAKER_MATCHING_ENABLED = (
    os.getenv(
        "TRUSTVOICE_ENABLE_SPEAKER_MATCHING",
        "true",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


# ============================================================
# OPTIONAL FAISS
# ============================================================

try:
    import faiss
except Exception:
    faiss = None


# ============================================================
# OPTIONAL HEAVY DEPENDENCIES
# ============================================================

torch = None
librosa = None
EncoderClassifier = None

_SPEAKER_IMPORT_ERROR: Exception | None = None
_MODEL = None

# Actual backend currently being used.
_SPEAKER_BACKEND = None


# ============================================================
# BASIC STATUS
# ============================================================

def is_speaker_matching_enabled() -> bool:
    return bool(SPEAKER_MATCHING_ENABLED)


def speaker_matching_status() -> dict[str, Any]:
    """
    Deployment-safe status.

    This function does not force-load SpeechBrain.
    """

    if not SPEAKER_MATCHING_ENABLED:
        return {
            "enabled": False,
            "available": False,
            "status": "DISABLED",
            "backend": None,
            "reason": (
                "Speaker matching is disabled for this deployment."
            ),
        }

    backend = _SPEAKER_BACKEND

    if backend:
        return {
            "enabled": True,
            "available": True,
            "status": "READY",
            "backend": backend,
            "reason": (
                "Speaker matching is available."
            ),
        }

    return {
        "enabled": True,
        "available": True,
        "status": "READY",
        "backend": (
            "ECAPA-TDNN"
            if EncoderClassifier is not None
            else "Local Acoustic Fingerprint"
        ),
        "reason": (
            "Speaker matching will use ECAPA-TDNN when "
            "available, otherwise the local acoustic "
            "fingerprint backend."
        ),
    }


# ============================================================
# REGISTRY
# ============================================================

def _ensure_registry() -> None:
    REGISTRY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not REGISTRY_FILE.exists():
        REGISTRY_FILE.write_text(
            json.dumps(
                {
                    "version": 2,
                    "speakers": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_registry() -> dict[str, Any]:
    _ensure_registry()

    try:
        data = json.loads(
            REGISTRY_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):
            raise ValueError(
                "Registry root must be an object."
            )

        data.setdefault(
            "version",
            2,
        )

        data.setdefault(
            "speakers",
            {},
        )

        if not isinstance(
            data["speakers"],
            dict,
        ):
            data["speakers"] = {}

        return data

    except Exception:
        return {
            "version": 2,
            "speakers": {},
        }


def save_registry(
    registry: dict[str, Any],
) -> None:
    _ensure_registry()

    temporary_file = REGISTRY_FILE.with_suffix(
        ".tmp"
    )

    temporary_file.write_text(
        json.dumps(
            registry,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_file.replace(
        REGISTRY_FILE
    )


# ============================================================
# LOAD LIBROSA
# ============================================================

def _load_librosa():
    global librosa

    if librosa is not None:
        return librosa

    try:
        import librosa as _librosa

        librosa = _librosa
        return librosa

    except Exception as exc:
        raise RuntimeError(
            "Librosa could not be loaded."
        ) from exc


# ============================================================
# TRY ECAPA / SPEECHBRAIN
# ============================================================

def _try_load_ecapa() -> bool:
    """
    Try loading ECAPA dependencies.

    Failure here is NOT fatal.
    The local acoustic backend will be used instead.
    """

    global torch
    global librosa
    global EncoderClassifier
    global _SPEAKER_IMPORT_ERROR
    global _SPEAKER_BACKEND

    if EncoderClassifier is not None:
        return True

    try:
        import torch as _torch

        torch = _torch

        _load_librosa()

        try:
            from speechbrain.inference.speaker import (
                EncoderClassifier as _EncoderClassifier
            )
        except Exception:
            from speechbrain.pretrained import (
                EncoderClassifier as _EncoderClassifier
            )

        EncoderClassifier = _EncoderClassifier

        _SPEAKER_BACKEND = "ECAPA-TDNN"

        return True

    except Exception as exc:
        _SPEAKER_IMPORT_ERROR = exc
        EncoderClassifier = None
        _SPEAKER_BACKEND = None
        return False


# ============================================================
# ECAPA MODEL
# ============================================================

def _get_model():
    global _MODEL

    if not SPEAKER_MATCHING_ENABLED:
        raise RuntimeError(
            "Speaker matching is disabled."
        )

    if not _try_load_ecapa():
        raise RuntimeError(
            "ECAPA-TDNN is unavailable."
        )

    if _MODEL is None:

        model_directory = (
            BASE_DIR
            / "models"
            / "ecapa_voxceleb"
        )

        model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        _MODEL = EncoderClassifier.from_hparams(
            source=(
                "speechbrain/"
                "spkrec-ecapa-voxceleb"
            ),
            savedir=str(
                model_directory
            ),
            run_opts={
                "device": "cpu"
            },
        )

    return _MODEL


# ============================================================
# AUDIO DECODING
# ============================================================

def decode_audio(
    raw: bytes,
    filename: str,
) -> np.ndarray:

    if not raw:
        raise ValueError(
            "Empty audio input."
        )

    lib = _load_librosa()

    suffix = (
        Path(filename).suffix.lower()
        or ".bin"
    )

    temporary_directory = None
    audio = None

    try:

        temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        work_dir = Path(
            temporary_directory.name
        )

        source_file = (
            work_dir
            / f"input{suffix}"
        )

        output_file = (
            work_dir
            / "speaker_16k.wav"
        )

        source_file.write_bytes(
            raw
        )

        try:
            import imageio_ffmpeg

            ffmpeg_exe = (
                imageio_ffmpeg
                .get_ffmpeg_exe()
            )
        except Exception as exc:
            raise RuntimeError(
                "imageio-ffmpeg is required "
                "for speaker audio decoding."
            ) from exc

        process = subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_file),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if (
            process.returncode != 0
            or not output_file.exists()
            or output_file.stat().st_size == 0
        ):
            detail = (
                process.stderr.strip()[-700:]
                or "Unsupported or corrupt audio."
            )

            raise ValueError(
                f"Could not decode audio: {detail}"
            )

        audio, _ = lib.load(
            str(output_file),
            sr=SAMPLE_RATE,
            mono=True,
        )

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            "Could not decode the audio "
            "for speaker matching."
        ) from exc

    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    audio = np.nan_to_num(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )

    if audio.size < SAMPLE_RATE:
        raise ValueError(
            "Audio must contain at least "
            "one second of speech."
        )

    # Remove silence where possible.
    try:
        intervals = lib.effects.split(
            audio,
            top_db=30,
        )
    except Exception:
        intervals = []

    if len(intervals):
        pieces = [
            audio[start:end]
            for start, end in intervals
            if end > start
        ]

        if pieces:
            audio = np.concatenate(
                pieces
            )

    if audio.size < SAMPLE_RATE:
        raise ValueError(
            "Not enough speech remains "
            "after silence removal."
        )

    # Limit to 30 seconds.
    max_samples = SAMPLE_RATE * 30

    if len(audio) > max_samples:

        start = (
            len(audio)
            - max_samples
        ) // 2

        audio = audio[
            start:
            start + max_samples
        ]

    # Normalize amplitude.
    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak > 1e-6:
        audio = (
            audio / peak
        ).astype(
            np.float32
        )

    return np.ascontiguousarray(
        audio,
        dtype=np.float32,
    )


# ============================================================
# LOCAL ACOUSTIC FINGERPRINT
# ============================================================

def _local_acoustic_embedding(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Lightweight speaker fingerprint.

    This is a deployment fallback when SpeechBrain/ECAPA
    cannot be imported.

    It is intentionally described as a similarity signal,
    not biometric identity proof.
    """

    lib = _load_librosa()

    if audio.size < SAMPLE_RATE:
        raise ValueError(
            "At least one second of speech is required."
        )

    features: list[np.ndarray] = []

    # MFCC
    mfcc = lib.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40,
        n_fft=512,
        hop_length=160,
    )

    features.append(
        np.mean(
            mfcc,
            axis=1,
        )
    )

    features.append(
        np.std(
            mfcc,
            axis=1,
        )
    )

    # Delta
    try:
        delta = lib.feature.delta(
            mfcc
        )

        features.append(
            np.mean(
                delta,
                axis=1,
            )
        )

        features.append(
            np.std(
                delta,
                axis=1,
            )
        )

    except Exception:
        pass

    # Spectral features
    spectral_centroid = (
        lib.feature.spectral_centroid(
            y=audio,
            sr=SAMPLE_RATE,
        )
    )

    spectral_bandwidth = (
        lib.feature.spectral_bandwidth(
            y=audio,
            sr=SAMPLE_RATE,
        )
    )

    spectral_rolloff = (
        lib.feature.spectral_rolloff(
            y=audio,
            sr=SAMPLE_RATE,
        )
    )

    zero_crossing = (
        lib.feature.zero_crossing_rate(
            audio
        )
    )

    for matrix in [
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing,
    ]:
        features.append(
            np.asarray(
                [
                    float(
                        np.mean(matrix)
                    ),
                    float(
                        np.std(matrix)
                    ),
                ],
                dtype=np.float32,
            )
        )

    # Chroma
    try:
        chroma = lib.feature.chroma_stft(
            y=audio,
            sr=SAMPLE_RATE,
        )

        features.append(
            np.mean(
                chroma,
                axis=1,
            )
        )

        features.append(
            np.std(
                chroma,
                axis=1,
            )
        )
    except Exception:
        pass

    vector = np.concatenate(
        [
            np.asarray(
                item,
                dtype=np.float32
            ).reshape(-1)
            for item in features
        ]
    )

    vector = np.nan_to_num(
        vector
    )

    # Deterministically normalize each feature.
    mean = float(
        np.mean(vector)
    )

    std = float(
        np.std(vector)
    )

    if std > 1e-8:
        vector = (
            vector - mean
        ) / std
    else:
        vector = (
            vector - mean
        )

    # Fixed 192-dimensional representation.
    if vector.size < EMBEDDING_DIM:

        repeats = int(
            np.ceil(
                EMBEDDING_DIM
                / vector.size
            )
        )

        vector = np.tile(
            vector,
            repeats,
        )[:EMBEDDING_DIM]

    else:

        vector = vector[
            :EMBEDDING_DIM
        ]

    norm = np.linalg.norm(
        vector
    )

    if norm <= 1e-8:
        raise RuntimeError(
            "Could not create a stable "
            "speaker fingerprint."
        )

    return (
        vector / norm
    ).astype(
        np.float32
    )


# ============================================================
# CREATE SPEAKER EMBEDDING
# ============================================================

def create_embedding(
    raw: bytes,
    filename: str,
) -> np.ndarray:

    if not SPEAKER_MATCHING_ENABLED:
        raise RuntimeError(
            "Speaker matching is disabled."
        )

    audio = decode_audio(
        raw,
        filename,
    )

    # Prefer ECAPA when it can actually be loaded.
    if _try_load_ecapa():

        try:
            model = _get_model()

            waveform = torch.tensor(
                audio,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():

                embedding = (
                    model.encode_batch(
                        waveform
                    )
                )

            embedding = (
                embedding
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

            if (
                embedding.size
                == EMBEDDING_DIM
            ):

                norm = np.linalg.norm(
                    embedding
                )

                if norm > 1e-8:

                    return (
                        embedding / norm
                    ).astype(
                        np.float32
                    )

        except Exception as exc:
            # ECAPA failure should NOT break
            # Voice Registry.
            global _SPEAKER_IMPORT_ERROR
            _SPEAKER_IMPORT_ERROR = exc

    # Deployment-safe fallback.
    global _SPEAKER_BACKEND

    _SPEAKER_BACKEND = (
        "Local Acoustic Fingerprint"
    )

    return _local_acoustic_embedding(
        audio
    )


# ============================================================
# FAISS
# ============================================================

def _require_faiss() -> None:

    if faiss is None:
        raise RuntimeError(
            "FAISS is unavailable."
        )


def _load_index():

    _require_faiss()

    if INDEX_FILE.exists():

        try:

            index = faiss.read_index(
                str(INDEX_FILE)
            )

            if (
                index.d
                != EMBEDDING_DIM
            ):
                raise RuntimeError(
                    "FAISS index has incompatible "
                    "embedding dimension."
                )

            return index

        except Exception:
            # Corrupt index is rebuilt from registry.
            pass

    return faiss.IndexFlatIP(
        EMBEDDING_DIM
    )


# ============================================================
# REBUILD INDEX
# ============================================================

def rebuild_index() -> int:

    if not SPEAKER_MATCHING_ENABLED:
        raise RuntimeError(
            "Speaker matching is disabled."
        )

    registry = load_registry()

    speakers = list(
        registry.get(
            "speakers",
            {},
        ).values()
    )

    if faiss is None:
        # Registry can still work without FAISS.
        # Return number of valid speakers.
        valid = 0

        for speaker in speakers:

            embedding = speaker.get(
                "embedding"
            )

            if isinstance(
                embedding,
                list,
            ) and len(embedding) == EMBEDDING_DIM:

                valid += 1

        return valid

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

        if (
            vector.size
            != EMBEDDING_DIM
        ):
            continue

        norm = np.linalg.norm(
            vector
        )

        if norm <= 1e-8:
            continue

        vector = (
            vector / norm
        ).astype(
            np.float32
        )

        valid_embeddings.append(
            vector
        )

    if valid_embeddings:

        matrix = np.asarray(
            valid_embeddings,
            dtype=np.float32,
        )

        faiss.normalize_L2(
            matrix
        )

        index.add(
            matrix
        )

    REGISTRY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    return int(
        index.ntotal
    )


# ============================================================
# ENROLL SPEAKER
# ============================================================

def enroll_speaker(
    name: str,
    samples: list[
        tuple[bytes, str]
    ],
) -> dict[str, Any]:

    if not SPEAKER_MATCHING_ENABLED:
        raise RuntimeError(
            "Speaker matching is disabled "
            "in this deployment."
        )

    name = str(
        name or ""
    ).strip()

    if not name:
        raise ValueError(
            "Speaker name is required."
        )

    if not samples:
        raise ValueError(
            "At least one voice sample "
            "is required."
        )

    embeddings = []
    sample_hashes = []

    for raw, filename in samples:

        if not raw:
            continue

        embedding = create_embedding(
            raw,
            filename,
        )

        embeddings.append(
            embedding
        )

        sample_hashes.append(
            hashlib.sha256(
                raw
            ).hexdigest()[:16]
        )

    if not embeddings:
        raise ValueError(
            "No valid voice samples were provided."
        )

    mean_embedding = np.mean(
        np.stack(
            embeddings
        ),
        axis=0,
    )

    norm = np.linalg.norm(
        mean_embedding
    )

    if norm <= 1e-8:
        raise RuntimeError(
            "Could not create a stable "
            "speaker profile."
        )

    mean_embedding = (
        mean_embedding / norm
    ).astype(
        np.float32
    )

    registry = load_registry()

    speakers = registry.setdefault(
        "speakers",
        {},
    )

    speaker_id = hashlib.sha256(
        name.lower().encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    existing = speakers.get(
        speaker_id
    )

    # Preserve existing profile metadata when re-enrolling.
    created_at = (
        existing.get(
            "created_at"
        )
        if existing
        else None
    )

    speaker_record = {
        "id": speaker_id,
        "name": name,
        "embedding": (
            mean_embedding.tolist()
        ),
        "samples": len(
            samples
        ),
        "sample_hashes": (
            sample_hashes
        ),
        "backend": (
            _SPEAKER_BACKEND
            or "Local Acoustic Fingerprint"
        ),
    }

    if created_at:
        speaker_record[
            "created_at"
        ] = created_at

    speakers[speaker_id] = (
        speaker_record
    )

    save_registry(
        registry
    )

    index_size = 0

    try:
        index_size = rebuild_index()
    except Exception:
        # Registry itself remains valid.
        index_size = len(
            speakers
        )

    return {
        "speaker_id": speaker_id,
        "name": name,
        "samples": len(
            samples
        ),
        "embedding_dimension": (
            EMBEDDING_DIM
        ),
        "index_size": (
            index_size
            or len(speakers)
        ),
        "backend": (
            _SPEAKER_BACKEND
            or "Local Acoustic Fingerprint"
        ),
    }


# ============================================================
# MATCH SPEAKER
# ============================================================

def match_speaker(
    raw: bytes,
    filename: str,
) -> dict[str, Any]:

    if not SPEAKER_MATCHING_ENABLED:

        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "speaker_id": None,
            "similarity": 0.0,
            "cosine_similarity": 0.0,
            "threshold": MATCH_THRESHOLD,
            "reason": (
                "Speaker matching is disabled "
                "for this deployment."
            ),
            "method": (
                "Speaker matching disabled"
            ),
            "registered_speakers": 0,
            "top_matches": [],
            "interpretation": (
                "Speaker similarity is an optional "
                "signal and is not proof of identity "
                "or voice authenticity."
            ),
        }

    registry = load_registry()

    speakers = list(
        registry.get(
            "speakers",
            {},
        ).values()
    )

    valid_speakers = []

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

        if (
            vector.size
            != EMBEDDING_DIM
        ):
            continue

        norm = np.linalg.norm(
            vector
        )

        if norm <= 1e-8:
            continue

        valid_speakers.append(
            (
                speaker,
                vector / norm,
            )
        )

    if not valid_speakers:

        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "speaker_id": None,
            "similarity": 0.0,
            "cosine_similarity": 0.0,
            "threshold": MATCH_THRESHOLD,
            "reason": (
                "No registered speakers."
            ),
            "method": (
                "Speaker matching"
            ),
            "registered_speakers": 0,
            "top_matches": [],
            "interpretation": (
                "Register at least one speaker "
                "before matching."
            ),
        }

    try:

        query = create_embedding(
            raw,
            filename,
        )

    except Exception as exc:

        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "speaker_id": None,
            "similarity": 0.0,
            "cosine_similarity": 0.0,
            "threshold": MATCH_THRESHOLD,
            "reason": (
                "Could not create speaker embedding. "
                f"Detail: {exc}"
            ),
            "method": (
                "Speaker matching unavailable"
            ),
            "registered_speakers": len(
                valid_speakers
            ),
            "top_matches": [],
            "interpretation": (
                "Speaker similarity is not "
                "identity proof."
            ),
        }

    query_norm = np.linalg.norm(
        query
    )

    if query_norm <= 1e-8:
        return {
            "available": False,
            "matched": False,
            "speaker": None,
            "speaker_id": None,
            "similarity": 0.0,
            "cosine_similarity": 0.0,
            "threshold": MATCH_THRESHOLD,
            "reason": (
                "Invalid speaker embedding."
            ),
            "method": (
                "Speaker matching unavailable"
            ),
            "registered_speakers": len(
                valid_speakers
            ),
            "top_matches": [],
            "interpretation": (
                "Speaker similarity is not "
                "identity proof."
            ),
        }

    query = (
        query / query_norm
    ).astype(
        np.float32
    )

    scored = []

    for speaker, vector in valid_speakers:

        similarity = float(
            np.clip(
                np.dot(
                    query,
                    vector,
                ),
                -1.0,
                1.0,
            )
        )

        scored.append(
            (
                similarity,
                speaker,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_similarity, best_speaker = (
        scored[0]
    )

    top_matches = []

    for similarity, speaker in scored[:3]:

        top_matches.append(
            {
                "speaker": speaker.get(
                    "name",
                    "Unknown",
                ),
                "similarity": round(
                    similarity * 100.0,
                    2,
                ),
            }
        )

    matched = (
        best_similarity
        >= MATCH_THRESHOLD
    )

    return {
        "available": True,
        "matched": matched,
        "speaker": (
            best_speaker.get(
                "name"
            )
            if matched
            else None
        ),
        "speaker_id": (
            best_speaker.get(
                "id"
            )
            if matched
            else None
        ),
        "similarity": round(
            float(
                np.clip(
                    best_similarity,
                    -1.0,
                    1.0,
                )
            ) * 100.0,
            2,
        ),
        "cosine_similarity": round(
            best_similarity,
            5,
        ),
        "threshold": MATCH_THRESHOLD,
        "method": (
            _SPEAKER_BACKEND
            or "Local Acoustic Fingerprint"
        ),
        "registered_speakers": len(
            valid_speakers
        ),
        "top_matches": top_matches,
        "reason": (
            "Speaker match found."
            if matched
            else "No speaker match above threshold."
        ),
        "interpretation": (
            "Speaker similarity is a matching "
            "signal only. It is not proof of "
            "identity and is not proof that "
            "the voice is authentic."
        ),
    }


# ============================================================
# LIST REGISTERED SPEAKERS
# ============================================================

def list_registered_speakers() -> list[
    dict[str, Any]
]:

    registry = load_registry()

    speakers = registry.get(
        "speakers",
        {},
    )

    result = []

    for speaker in speakers.values():

        result.append(
            {
                "id": speaker.get(
                    "id"
                ),
                "name": speaker.get(
                    "name"
                ),
                "samples": speaker.get(
                    "samples",
                    0,
                ),
                "backend": speaker.get(
                    "backend",
                    "Unknown",
                ),
            }
        )

    return result


# ============================================================
# DELETE SPEAKER
# ============================================================

def delete_speaker(
    speaker_id: str,
) -> bool:

    if not SPEAKER_MATCHING_ENABLED:
        return False

    registry = load_registry()

    speakers = registry.get(
        "speakers",
        {},
    )

    if speaker_id not in speakers:
        return False

    del speakers[
        speaker_id
    ]

    save_registry(
        registry
    )

    try:
        rebuild_index()
    except Exception:
        pass

    return True


# ============================================================
# DIRECT SPEAKER COMPARISON
# ============================================================

def compare_speaker_audio(
    reference_raw: bytes,
    reference_name: str,
    target_raw: bytes,
    target_name: str,
) -> dict[str, Any]:

    if not SPEAKER_MATCHING_ENABLED:

        return {
            "available": False,
            "score": 0.0,
            "cosine": 0.0,
            "reference_file": reference_name,
            "target_file": target_name,
            "reason": (
                "Speaker matching is disabled "
                "for this deployment."
            ),
            "method": (
                "Speaker comparison disabled"
            ),
            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    try:

        reference_embedding = (
            create_embedding(
                reference_raw,
                reference_name,
            )
        )

        target_embedding = (
            create_embedding(
                target_raw,
                target_name,
            )
        )

        reference_embedding = (
            reference_embedding
            / max(
                np.linalg.norm(
                    reference_embedding
                ),
                1e-8,
            )
        )

        target_embedding = (
            target_embedding
            / max(
                np.linalg.norm(
                    target_embedding
                ),
                1e-8,
            )
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
                _SPEAKER_BACKEND
                or "Local Acoustic Fingerprint"
            ),
            "interpretation": (
                "Similarity signal only. "
                "It is not proof of speaker identity "
                "or voice authenticity."
            ),
        }

    except Exception as exc:

        return {
            "available": False,
            "score": 0.0,
            "cosine": 0.0,
            "reference_file": reference_name,
            "target_file": target_name,
            "reason": (
                "Speaker comparison failed. "
                f"Detail: {exc}"
            ),
            "method": (
                "Speaker comparison unavailable"
            ),
            "interpretation": (
                "Similarity signal only. "
                "It is not proof of speaker identity "
                "or voice authenticity."
            ),
        }


# ============================================================
# DEPLOYMENT SELF TEST
# ============================================================

def deployment_self_test() -> dict[str, Any]:

    return {
        "module_imported": True,
        "speaker_matching_enabled": (
            SPEAKER_MATCHING_ENABLED
        ),
        "faiss_available": (
            faiss is not None
        ),
        "registry_directory": str(
            REGISTRY_DIR
        ),
        "registry_exists": (
            REGISTRY_FILE.exists()
        ),
        "index_exists": (
            INDEX_FILE.exists()
        ),
        "speaker_matching_status": (
            speaker_matching_status()
        ),
        "backend": (
            _SPEAKER_BACKEND
            or (
                "ECAPA-TDNN"
                if EncoderClassifier is not None
                else "Local Acoustic Fingerprint"
            )
        ),
    }