class ConfigurationError(ValueError):
    pass


class VoiceNotFound(FileNotFoundError):
    pass


class ReferenceInvalid(ValueError):
    pass


class GenerationError(RuntimeError):
    pass


class AudioValidationError(ValueError):
    pass


class TranscriptQCError(RuntimeError):
    pass


class OutOfMemory(GenerationError):
    pass
