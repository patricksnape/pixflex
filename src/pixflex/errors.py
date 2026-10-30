"""Exception types raised by pixflex."""


class PixflexError(Exception):
    """Base class for all pixflex errors."""


class ShapeMismatch(PixflexError, ValueError):
    """Content does not match its slot and the slot's fit is ``"exact"``."""


class DuplicateKeyError(PixflexError, ValueError):
    """Two nodes in one layout share a key."""


class ForeignSlotError(PixflexError, KeyError):
    """A handle from a different layout was used with this plan or canvas."""


class ChannelMismatch(PixflexError, ValueError):
    """Content and canvas disagree about the number of channels."""


class DTypeMismatch(PixflexError, TypeError):
    """Content dtype cannot be safely converted to the canvas dtype."""


class DeviceMismatch(PixflexError, ValueError):
    """Content lives on a different device than the canvas."""


class BackendMismatch(PixflexError, TypeError):
    """Content comes from a different array library than the canvas."""


class MissingResizer(PixflexError, RuntimeError):
    """Resizing is needed but no resizer is available for the array library."""
