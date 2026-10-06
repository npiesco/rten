# RTen model format

This vendored runtime reads one current `.rten` representation. Its test
builder and loader change together. There is no generation field, headerless
decoder, or compatibility promise for upstream converter output. Fogscrib's
recognizer loads the original ONNX models directly.

The `rten_format` feature exposes the current `ModelBuilder`, `GraphBuilder`,
`OpType` and `MetadataArgs` producer APIs. The runtime integration test invokes
that builder and executes its serialized graph through `Model::load`.

## Framing

```text
[magic:u8x4] [model_offset:u64] [model_len:u64] [tensor_data_offset:u64]
```

The 28-byte header starts with the ASCII discriminator `RTEN`. All integers
are little-endian. Offsets must be within the file; the model starts after
the header, its checked length must fit, and tensor data cannot overlap it.

## Graph and weights

The FlatBuffers root has two fields, in order: required `graph:Graph` and
optional `metadata:Metadata`. The graph, metadata, node and operator types
come from `rten-model-file::schema`. `src/model/rten_format.rs` implements
and verifies the current root; it does not use the dependency's versioned root
or header. FlatBuffers verification still covers every reachable graph field.

Constants require a type, shape and byte offset relative to the tensor segment.
Inline constant records are not accepted. Checked shape multiplication, byte
length and offset addition reject out-of-range weights before tensor creation.
Misaligned data is copied into aligned tensor storage.

Attribute-bearing operators require their attribute record, including CumSum,
Pad and Shape. Optional values inside the current record retain their operator
semantics; an absent record is not treated as an older encoding.

Run `cargo test --manifest-path vendor/rten/Cargo.toml --test current_format`
from the repository root to execute real serialized models through the loader
and disk path and exercise malformed framing and weights.
