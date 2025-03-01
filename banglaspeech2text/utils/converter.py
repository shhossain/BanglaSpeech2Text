# !ct2-transformers-converter --model anuragshas/whisper-large-v2-bn --output_dir whisper-large-v2-ct2 --quantization float16

from pathlib import Path
import subprocess
import logging

# Get a child logger that inherits from the main logger
logger = logging.getLogger("BanglaSpeech2Text.converter")


def is_ct2_transformers_converter_available() -> bool:
    """Check if ct2-transformers-converter command is available."""
    try:
        subprocess.run(
            ["ct2-transformers-converter", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def convert_model(
    model_name: str, output_dir: str, compute_type: str = "float16"
) -> bool:
    """
    Convert transformers model to CTranslate2 format.

    Args:
        model_name: The Hugging Face model name or path
        output_dir: Output directory for the converted model
        compute_type: Quantization type (float16, int8, int8_float16, etc.)

    Returns:
        bool: True if conversion succeeded
    """
    if not is_ct2_transformers_converter_available():
        logger.error(
            "faster-whisper is not installed correctly. Please install it again with `pip install faster-whisper --force-reinstall`."
        )
        return False

    try:
        cmd = [
            "ct2-transformers-converter",
            "--model",
            model_name,
            "--output_dir",
            output_dir,
            "--quantization",
            compute_type,
        ]

        logger.info(f"Converting model {model_name} to CTranslate2 format...")
        logger.info(f"Command: {' '.join(cmd)}")

        subprocess.check_call(cmd)
        logger.info(f"Successfully converted model to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting model: {e}")
        return False


def get_ct2_model_path(
    model_name: str, cache_dir: Path, compute_type: str = "float16"
) -> str:
    """
    Get path to CTranslate2 model, converting if necessary.

    Args:
        model_name: The Hugging Face model name
        cache_dir: Cache directory for storing converted models
        quantcompute_typeization: Quantization type (float16, int8, int8_float16)

    Returns:
        str: Path to the converted model
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Format model name for file system
    safe_model_name = model_name.replace("/", "--")
    ct2_dir_name = f"{safe_model_name}-ct2-{compute_type}"
    ct2_model_path = cache_dir / ct2_dir_name

    # Check if model already exists
    if ct2_model_path.exists():
        logger.info(f"Found existing CTranslate2 model at {ct2_model_path}")
        return str(ct2_model_path)

    # Convert model
    logger.info(f"CTranslate2 model not found at {ct2_model_path}. Converting...")
    if convert_model(model_name, str(ct2_model_path), compute_type):
        return str(ct2_model_path)
    else:
        raise RuntimeError(
            f"Failed to convert model {model_name} to CTranslate2 format"
        )
