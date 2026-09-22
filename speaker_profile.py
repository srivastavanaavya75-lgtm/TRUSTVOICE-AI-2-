"""
TrustVoice AI
Registered Speaker Profile + Optional Speaker Matching

IMPORTANT:
Speaker matching is OPTIONAL.

By default:
    TRUSTVOICE_ENABLE_SPEAKER_MATCHING=false

This means:
    - Audio Forensics works without SpeechBrain
    - AASIST works independently
    - Whisper works independently
    - Risk Engine works independently
    - Streamlit Cloud does NOT load SpeechBrain during normal analysis

To explicitly enable speaker matching in a compatible environment:

    TRUSTVOICE_ENABLE_SPEAKER_MATCHING=true

Speaker similarity is NOT identity proof and is NOT voice-authenticity proof.
"""

from __future__ import annotations

import hashlib
import json
import os
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
# SPEAKER MATCHING FEATURE FLAG
# ============================================================
#
# DEFAULT = FALSE
#
# This is the critical deployment protection.
#
# Audio Forensics will NOT load:
#   torch
#   librosa
#   speechbrain
#   flair
#   spacy
#
# unless speaker matching is explicitly enabled.
# ============================================================

SPEAKER_MATCHING_ENABLED = (
    os.getenv(
        "TRUSTVOICE_ENABLE_SPEAKER_MATCHING",
        "false",
    ).strip().lower()
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
# HEAVY DEPENDENCIES
# ============================================================
#
# NEVER import these at module level.
#
# They are loaded only if speaker matching is explicitly enabled.
# ============================================================

torch = None
librosa = None
EncoderClassifier = None

_SPEAKER_IMPORT_ERROR: Exception | None = None

_MODEL = None


# ============================================================
# DEPLOYMENT STATUS
# ============================================================

def is_speaker_matching_enabled() -> bool:
    """
    Return whether optional speaker matching is enabled.
    """

    return bool(
        SPEAKER_MATCHING_ENABLED
    )


def speaker_matching_status() -> dict[str, Any]:
    """
    Return deployment-safe speaker matching status.
    """

    if not SPEAKER_MATCHING_ENABLED:

        return {
            "enabled": False,
            "available": False,
            "status": "DISABLED",
            "reason": (
                "Speaker matching is disabled for "
                "this deployment. Core voice authenticity "
                "and risk analysis continue independently."
            ),
        }

    if _SPEAKER_IMPORT_ERROR is not None:

        return {
            "enabled": True,
            "available": False,
            "status": "UNAVAILABLE",
            "reason": str(
                _SPEAKER_IMPORT_ERROR
            ),
        }

    return {
        "enabled": True,
        "available": (
            EncoderClassifier is not None
            and torch is not None
            and librosa is not None
        ),
        "status": (
            "READY"
            if EncoderClassifier is not None
            else "NOT_LOADED"
        ),
        "reason": (
            "Speaker matching dependencies "
            "will be loaded only when required."
        ),
    }


# ============================================================
# REGISTRY DIRECTORY
# ============================================================

def _ensure_registry() -> None:
    """
    Ensure speaker registry directory exists.
    """

    REGISTRY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not REGISTRY_FILE.exists():

        REGISTRY_FILE.write_text(
            json.dumps(
                {
                    "version": 1,
                    "speakers": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )


# ============================================================
# LOAD REGISTRY
# ============================================================

def load_registry() -> dict[str, Any]:
    """
    Load registered speaker metadata.

    Returns an empty registry if the file is missing
    or corrupted.
    """

    _ensure_registry()

    try:

        data = json.loads(
            REGISTRY_FILE.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Registry root must be an object."
            )

        data.setdefault(
            "version",
            1,
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
            "version": 1,
            "speakers": {},
        }


# ============================================================
# SAVE REGISTRY
# ============================================================

def save_registry(
    registry: dict[str, Any],
) -> None:
    """
    Save speaker registry.
    """

    _ensure_registry()

    REGISTRY_FILE.write_text(
        json.dumps(
            registry,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# OPTIONAL DEPENDENCY LOADER
# ============================================================

def _load_optional_speaker_dependencies() -> None:
    """
    Lazily load Torch, Librosa and SpeechBrain.

    This function must NEVER execute during normal
    Audio Forensics analysis because speaker matching
    is disabled by default.
    """

    global torch
    global librosa
    global EncoderClassifier
    global _SPEAKER_IMPORT_ERROR

    if not SPEAKER_MATCHING_ENABLED:

        raise RuntimeError(
            "Speaker matching is disabled."
        )

    if _SPEAKER_IMPORT_ERROR is not None:

        raise RuntimeError(
            "Speaker matching dependencies "
            "are unavailable. "
            f"Detail: {_SPEAKER_IMPORT_ERROR}"
        )

    if (
        torch is not None
        and librosa is not None
        and EncoderClassifier is not None
    ):

        return

    try:

        # ----------------------------------------------------
        # Torch
        # ----------------------------------------------------

        import torch as _torch

        # ----------------------------------------------------
        # Librosa
        # ----------------------------------------------------

        import librosa as _librosa

        # ----------------------------------------------------
        # SpeechBrain
        # ----------------------------------------------------

        try:

            from speechbrain.inference.speaker import (
                EncoderClassifier as _EncoderClassifier
            )

        except Exception:

            try:

                from speechbrain.pretrained import (
                    EncoderClassifier as _EncoderClassifier
                )

            except Exception as speechbrain_error:

                raise RuntimeError(
                    "SpeechBrain could not be loaded."
                ) from speechbrain_error

        # ----------------------------------------------------
        # Save imports
        # ----------------------------------------------------

        torch = _torch

        librosa = _librosa

        EncoderClassifier = _EncoderClassifier

    except Exception as exc:

        _SPEAKER_IMPORT_ERROR = exc

        raise RuntimeError(
            "Optional speaker-recognition dependencies "
            "could not be loaded. "
            "Core TrustVoice analysis can continue "
            "without speaker matching."
        ) from exc


# ============================================================
# REQUIRE SPEAKER MODELS
# ============================================================

def _require_models() -> None:
    """
    Ensure optional speaker dependencies are loaded.
    """

    if not SPEAKER_MATCHING_ENABLED:

        raise RuntimeError(
            "Speaker matching is disabled."
        )

    _load_optional_speaker_dependencies()


# ============================================================
# LOAD ECAPA MODEL
# ============================================================

def _get_model():
    """
    Lazily load SpeechBrain ECAPA speaker model.
    """

    global _MODEL

    if not SPEAKER_MATCHING_ENABLED:

        raise RuntimeError(
            "Speaker matching is disabled."
        )

    _load_optional_speaker_dependencies()

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

        _MODEL = (
            EncoderClassifier.from_hparams(
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
        )

    return _MODEL


# ============================================================
# AUDIO DECODING
# ============================================================

def decode_audio(
    raw: bytes,
    filename: str,
) -> np.ndarray:
    """
    Decode an audio file into:

        mono
        16 kHz
        float32

    This function is ONLY used by speaker matching.
    """

    _require_models()

    if not raw:

        raise ValueError(
            "Empty audio input."
        )

    suffix = (
        Path(filename).suffix.lower()
        or ".bin"
    )

    temporary_directory = None

    try:

        import subprocess
        import tempfile

        import imageio_ffmpeg

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

        ffmpeg_exe = (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

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

        audio, _ = librosa.load(
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

        if (
            temporary_directory
            is not None
        ):

            temporary_directory.cleanup()

    # --------------------------------------------------------
    # Numerical cleanup
    # --------------------------------------------------------

    audio = np.nan_to_num(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )

    # --------------------------------------------------------
    # Minimum duration
    # --------------------------------------------------------

    if audio.size < SAMPLE_RATE:

        raise ValueError(
            "Audio must contain at least "
            "one second of speech."
        )

    # --------------------------------------------------------
    # Silence removal
    # --------------------------------------------------------

    try:

        intervals = (
            librosa.effects.split(
                audio,
                top_db=30,
            )
        )

    except Exception:

        intervals = []

    if len(intervals):

        pieces = []

        for start, end in intervals:

            pieces.append(
                audio[
                    start:end
                ]
            )

        if pieces:

            audio = np.concatenate(
                pieces
            )

    if audio.size < SAMPLE_RATE:

        raise ValueError(
            "Not enough speech remains "
            "after silence removal."
        )

    # --------------------------------------------------------
    # Limit to 30 seconds
    # --------------------------------------------------------

    max_samples = (
        SAMPLE_RATE * 30
    )

    if len(audio) > max_samples:

        start = (
            len(audio)
            - max_samples
        ) // 2

        audio = audio[
            start:
            start + max_samples
        ]

    return np.ascontiguousarray(
        audio,
        dtype=np.float32,
    )


# ============================================================
# CREATE SPEAKER EMBEDDING
# ============================================================

def create_embedding(
    raw: bytes,
    filename: str,
) -> np.ndarray:
    """
    Create a normalized ECAPA speaker embedding.
    """

    if not SPEAKER_MATCHING_ENABLED:

        raise RuntimeError(
            "Speaker matching is disabled."
        )

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
        != EMBEDDING_DIM
    ):

        raise RuntimeError(
            "Unexpected speaker embedding "
            f"dimension: {embedding.size}. "
            f"Expected: {EMBEDDING_DIM}."
        )

    norm = np.linalg.norm(
        embedding
    )

    if norm <= 0:

        raise RuntimeError(
            "Speaker embedding has zero norm."
        )

    return (
        embedding / norm
    ).astype(np.float32)


# ============================================================
# FAISS CHECK
# ============================================================

def _require_faiss() -> None:
    """
    Check whether FAISS is available.
    """

    if faiss is None:

        raise RuntimeError(
            "FAISS is unavailable."
        )


# ============================================================
# LOAD / CREATE FAISS INDEX
# ============================================================

def _load_index():
    """
    Load the existing FAISS speaker index.

    Creates an empty index when no index exists.
    """

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

        except Exception as exc:

            raise RuntimeError(
                "Could not load the speaker FAISS index."
            ) from exc

    return faiss.IndexFlatIP(
        EMBEDDING_DIM
    )


# ============================================================
# REBUILD FAISS INDEX
# ============================================================

def rebuild_index() -> int:
    """
    Rebuild FAISS index from registered speakers.
    """

    if not SPEAKER_MATCHING_ENABLED:

        raise RuntimeError(
            "Speaker matching is disabled."
        )

    _require_faiss()

    registry = load_registry()

    speakers = list(
        registry
        .get(
            "speakers",
            {},
        )
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

        if (
            vector.size
            != EMBEDDING_DIM
        ):

            continue

        norm = np.linalg.norm(
            vector
        )

        if norm <= 0:

            continue

        vector = (
            vector / norm
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
    """
    Register a speaker using one or more audio samples.
    """

    if not SPEAKER_MATCHING_ENABLED:

        raise RuntimeError(
            "Speaker matching is disabled "
            "in this deployment."
        )

    _require_faiss()

    name = name.strip()

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

    mean_embedding = np.mean(
        np.stack(
            embeddings
        ),
        axis=0,
    )

    norm = np.linalg.norm(
        mean_embedding
    )

    if norm <= 0:

        raise RuntimeError(
            "Could not create a stable "
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
        name.lower().encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    speakers[speaker_id] = {

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
    }

    save_registry(
        registry
    )

    rebuild_index()

    return {

        "speaker_id": speaker_id,

        "name": name,

        "samples": len(
            samples
        ),

        "embedding_dimension": (
            EMBEDDING_DIM
        ),

        "index_size": len(
            speakers
        ),
    }


# ============================================================
# MATCH SPEAKER
# ============================================================

def match_speaker(
    raw: bytes,
    filename: str,
) -> dict[str, Any]:
    """
    Match an audio sample against registered speakers.

    IMPORTANT:
    Speaker matching is disabled by default.

    When disabled, this function returns immediately.
    Therefore Audio Forensics will NOT load SpeechBrain,
    Torch, Librosa, Flair or spaCy.
    """

    # ========================================================
    # HARD DEPLOYMENT GUARD
    # ========================================================
    #
    # This MUST remain the first executable block.
    #
    # Normal Audio Forensics analysis reaches this function,
    # but exits here without loading any speaker dependencies.
    # ========================================================

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
                "for this deployment. "
                "Voice authenticity and risk analysis "
                "continue independently."
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

    # ========================================================
    # OPTIONAL DEPENDENCY LOADING
    # ========================================================

    try:

        _load_optional_speaker_dependencies()

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
                "Speaker matching dependencies "
                "are unavailable. "
                "Core TrustVoice analysis can continue. "
                f"Detail: {exc}"
            ),

            "method": (
                "Speaker matching unavailable"
            ),

            "registered_speakers": 0,

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    # ========================================================
    # FAISS
    # ========================================================

    try:

        _require_faiss()

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
                "FAISS is unavailable. "
                f"Detail: {exc}"
            ),

            "method": (
                "Speaker matching unavailable"
            ),

            "registered_speakers": 0,

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    # ========================================================
    # REGISTRY
    # ========================================================

    registry = load_registry()

    speakers = list(
        registry
        .get(
            "speakers",
            {},
        )
        .values()
    )

    if not speakers:

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
                "ECAPA-TDNN + FAISS"
            ),

            "registered_speakers": 0,

            "top_matches": [],

            "interpretation": (
                "No speaker profile is available "
                "for comparison."
            ),
        }

    # ========================================================
    # LOAD INDEX
    # ========================================================

    try:

        index = _load_index()

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
                "Speaker index could not be loaded. "
                f"Detail: {exc}"
            ),

            "method": (
                "Speaker matching unavailable"
            ),

            "registered_speakers": len(
                speakers
            ),

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    # ========================================================
    # INDEX / REGISTRY CONSISTENCY
    # ========================================================

    if (
        index.ntotal
        != len(speakers)
    ):

        try:

            rebuild_index()

            index = _load_index()

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
                    "Speaker index rebuild failed. "
                    f"Detail: {exc}"
                ),

                "method": (
                    "Speaker matching unavailable"
                ),

                "registered_speakers": len(
                    speakers
                ),

                "top_matches": [],

                "interpretation": (
                    "Speaker similarity is not "
                    "identity proof or authenticity proof."
                ),
            }

    if index.ntotal == 0:

        return {

            "available": False,

            "matched": False,

            "speaker": None,

            "speaker_id": None,

            "similarity": 0.0,

            "cosine_similarity": 0.0,

            "threshold": MATCH_THRESHOLD,

            "reason": (
                "Speaker index contains "
                "no valid embeddings."
            ),

            "method": (
                "Speaker matching unavailable"
            ),

            "registered_speakers": len(
                speakers
            ),

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    try:

        query_embedding = (
            create_embedding(
                raw,
                filename,
            )
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
                speakers
            ),

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    # ========================================================
    # NORMALIZE QUERY
    # ========================================================

    query = (
        query_embedding
        .reshape(
            1,
            -1,
        )
        .astype(
            np.float32
        )
    )

    faiss.normalize_L2(
        query
    )

    # ========================================================
    # SEARCH
    # ========================================================

    try:

        similarities, indices = (
            index.search(
                query,
                min(
                    3,
                    index.ntotal,
                ),
            )
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
                "FAISS speaker search failed. "
                f"Detail: {exc}"
            ),

            "method": (
                "Speaker matching unavailable"
            ),

            "registered_speakers": len(
                speakers
            ),

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    if (
        len(indices) == 0
        or len(indices[0]) == 0
    ):

        return {

            "available": True,

            "matched": False,

            "speaker": None,

            "speaker_id": None,

            "similarity": 0.0,

            "cosine_similarity": 0.0,

            "threshold": MATCH_THRESHOLD,

            "reason": (
                "No speaker match found."
            ),

            "method": (
                "ECAPA-TDNN + FAISS"
            ),

            "registered_speakers": len(
                speakers
            ),

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    # ========================================================
    # BEST MATCH
    # ========================================================

    best_similarity = float(
        similarities[0][0]
    )

    best_index = int(
        indices[0][0]
    )

    if (
        best_index < 0
        or best_index >= len(
            speakers
        )
    ):

        return {

            "available": True,

            "matched": False,

            "speaker": None,

            "speaker_id": None,

            "similarity": 0.0,

            "cosine_similarity": 0.0,

            "threshold": MATCH_THRESHOLD,

            "reason": (
                "No valid speaker vector match."
            ),

            "method": (
                "ECAPA-TDNN + FAISS"
            ),

            "registered_speakers": len(
                speakers
            ),

            "top_matches": [],

            "interpretation": (
                "Speaker similarity is not "
                "identity proof or authenticity proof."
            ),
        }

    matched_speaker = speakers[
        best_index
    ]

    # ========================================================
    # TOP MATCHES
    # ========================================================

    top_matches = []

    for score, vector_index in zip(
        similarities[0],
        indices[0],
    ):

        vector_index = int(
            vector_index
        )

        if (
            vector_index >= 0
            and vector_index
            < len(speakers)
        ):

            top_matches.append(
                {
                    "speaker": speakers[
                        vector_index
                    ].get(
                        "name",
                        "Unknown",
                    ),
                    "similarity": round(
                        float(score)
                        * 100.0,
                        2,
                    ),
                }
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "available": True,

        "matched": (
            best_similarity
            >= MATCH_THRESHOLD
        ),

        "speaker": matched_speaker.get(
            "name"
        ),

        "speaker_id": matched_speaker.get(
            "id"
        ),

        "similarity": round(
            float(
                np.clip(
                    best_similarity,
                    -1.0,
                    1.0,
                )
            )
            * 100.0,
            2,
        ),

        "cosine_similarity": round(
            best_similarity,
            5,
        ),

        "threshold": (
            MATCH_THRESHOLD
        ),

        "method": (
            "ECAPA-TDNN speaker embedding "
            "+ FAISS cosine similarity"
        ),

        "registered_speakers": len(
            speakers
        ),

        "top_matches": top_matches,

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
    """
    Return registered speaker metadata.
    """

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
            }
        )

    return result


# ============================================================
# DELETE SPEAKER
# ============================================================

def delete_speaker(
    speaker_id: str,
) -> bool:
    """
    Delete a registered speaker.
    """

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
    """
    Compare two audio files using ECAPA embeddings.

    This function is also optional and disabled when
    speaker matching is disabled.
    """

    if not SPEAKER_MATCHING_ENABLED:

        return {

            "available": False,

            "score": 0.0,

            "cosine": 0.0,

            "reference_file": (
                reference_name
            ),

            "target_file": (
                target_name
            ),

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

            "reference_file": (
                reference_name
            ),

            "target_file": (
                target_name
            ),

            "method": (
                "ECAPA-TDNN speaker embeddings "
                "+ cosine similarity"
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

            "reference_file": (
                reference_name
            ),

            "target_file": (
                target_name
            ),

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
# DEPLOYMENT SELF-TEST
# ============================================================

def deployment_self_test() -> dict[str, Any]:
    """
    Lightweight deployment check.

    IMPORTANT:
    This function does NOT load SpeechBrain when
    speaker matching is disabled.
    """

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
    }


# ============================================================
# END
# ============================================================