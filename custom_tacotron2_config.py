from dataclasses import dataclass, field
from TTS.tts.configs.tacotron_config import TacotronConfig


@dataclass
class Tacotron2Config(TacotronConfig):
    """Defines parameters for Tacotron2 based models.

    Example:

        >>> from TTS.tts.configs.tacotron2_config import Tacotron2Config
        >>> config = Tacotron2Config()

    Check `TacotronConfig` for argument descriptions.
    """

    model: str = "tacotron2"
    out_channels: int = 80
    encoder_in_features: int = 512
    decoder_in_features: int = 512
    
    # NEW - Implemented variables for loss
    infoNCE_alpha: float = field(
        default=None,
        metadata={"help": "[More information needed]. Defaults to None"},
    )
    similarity_loss_alpha: float = field(
        default=None,
        metadata={"help": "[More information needed]. Defaults to None"},
    )
    return_wav: bool = field(
        default=False,
        metadata={"help": "[More information needed]. Defaults to False"},
    )