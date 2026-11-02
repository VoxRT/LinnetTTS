//! Error type of the runtime.

use std::fmt;

/// Everything that can go wrong while loading or driving a model.
#[derive(Debug)]
pub enum Error {
    /// File could not be read.
    Io(std::io::Error),
    /// The byte stream is not a container this runtime understands
    /// (safetensors header, SentencePiece model, `.npy` file).
    Format(String),
    /// The container parsed but the model inside is not the expected
    /// topology (missing tensor, wrong shape or dtype, bad config value).
    Shape(String),
    /// A runtime argument is out of range (voice latent width, empty
    /// input, KV cache overflow).
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Format(s) => write!(f, "format: {s}"),
            Error::Shape(s) => write!(f, "shape: {s}"),
            Error::Invalid(s) => write!(f, "invalid argument: {s}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// `Result` with this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
