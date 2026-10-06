//! Current RTen framing and root table. Graph/operator types come from the
//! published schema; the root contains only graph and metadata.

use std::error::Error;
use std::fmt::{Display, Formatter};

use flatbuffers::{FlatBufferBuilder, WIPOffset};
use flatbuffers::{Follow, ForwardsUOffset, Table, Verifiable};
use rten_model_file::schema as sg;

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub model_offset: u64,
    pub model_len: u64,
    pub tensor_data_offset: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HeaderError {
    TooShort,
    InvalidMagic,
    InvalidOffset,
    InvalidLength,
}

impl Header {
    pub const LEN: usize = 28;

    pub fn from_buf(buf: &[u8]) -> Result<Self, HeaderError> {
        let magic = buf.get(..4).ok_or(HeaderError::TooShort)?;
        if magic != b"RTEN" {
            return Err(HeaderError::InvalidMagic);
        }
        let data = buf.get(..Self::LEN).ok_or(HeaderError::TooShort)?;
        let read = |offset| u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let header = Self {
            model_offset: read(4),
            model_len: read(12),
            tensor_data_offset: read(20),
        };
        let file_size = buf.len() as u64;
        if header.model_offset < Self::LEN as u64 || header.model_offset > file_size {
            return Err(HeaderError::InvalidOffset);
        }
        let end = header
            .model_offset
            .checked_add(header.model_len)
            .filter(|end| *end <= file_size)
            .ok_or(HeaderError::InvalidLength)?;
        if header.tensor_data_offset < end || header.tensor_data_offset > file_size {
            return Err(HeaderError::InvalidOffset);
        }
        Ok(header)
    }

    pub fn to_buf(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(Self::LEN);
        buffer.extend(b"RTEN");
        buffer.extend(self.model_offset.to_le_bytes());
        buffer.extend(self.model_len.to_le_bytes());
        buffer.extend(self.tensor_data_offset.to_le_bytes());
        buffer
    }
}

impl Display for HeaderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TooShort => "header is too short",
            Self::InvalidMagic => "incorrect file magic",
            Self::InvalidOffset => "segment offset is invalid",
            Self::InvalidLength => "segment length is invalid",
        })
    }
}

impl Error for HeaderError {}

pub struct Model<'a> {
    table: Table<'a>,
}

impl<'a> Follow<'a> for Model<'a> {
    type Inner = Self;

    unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
        Self {
            // SAFETY: The caller guarantees a valid table at loc, as required by Follow.
            table: unsafe { Table::new(buf, loc) },
        }
    }
}

impl Verifiable for Model<'_> {
    fn run_verifier(
        verifier: &mut flatbuffers::Verifier,
        pos: usize,
    ) -> Result<(), flatbuffers::InvalidFlatbuffer> {
        verifier
            .visit_table(pos)?
            .visit_field::<ForwardsUOffset<sg::Graph<'_>>>("graph", 4, true)?
            .visit_field::<ForwardsUOffset<sg::Metadata<'_>>>("metadata", 6, false)?
            .finish();
        Ok(())
    }
}

impl<'a> Model<'a> {
    pub fn graph(&self) -> sg::Graph<'a> {
        // SAFETY: root_as_model verifies the required graph field and its full graph.
        unsafe { self.table.get::<ForwardsUOffset<sg::Graph<'a>>>(4, None) }.unwrap()
    }

    pub fn metadata(&self) -> Option<sg::Metadata<'a>> {
        // SAFETY: root_as_model verifies the optional metadata field when present.
        unsafe { self.table.get::<ForwardsUOffset<sg::Metadata<'a>>>(6, None) }
    }
}

pub fn root_as_model(data: &[u8]) -> Result<Model<'_>, flatbuffers::InvalidFlatbuffer> {
    flatbuffers::root::<Model<'_>>(data)
}

pub fn create_model<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    graph: WIPOffset<sg::Graph<'a>>,
    metadata: Option<WIPOffset<sg::Metadata<'a>>>,
) -> WIPOffset<Model<'a>> {
    let start = builder.start_table();
    builder.push_slot_always(4, graph);
    if let Some(metadata) = metadata {
        builder.push_slot_always(6, metadata);
    }
    let root = builder.end_table(start);
    WIPOffset::new(root.value())
}
