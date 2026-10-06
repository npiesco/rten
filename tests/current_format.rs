use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use flatbuffers::FlatBufferBuilder;
use rten::{DataType, Dimension, MetadataArgs, ModelBuilder, OpType};
use rten::{LoadErrorKind, Model, ModelOptions, RunOptions, ShapeInferenceMode, ThreadPool};
use rten_model_file::schema as sg;
use rten_tensor::Tensor;
use rten_tensor::prelude::*;

#[path = "../src/model/rten_format.rs"]
mod rten_format;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[test]
fn post_load_prepacking_preserves_real_matmul_outputs_with_an_explicit_pool() {
    let mut builder = ModelBuilder::new();
    let mut graph = builder.graph_builder();
    let weights = graph.add_constant(
        Tensor::from([[1_f32, 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]).view(),
    );
    let input = graph.add_value(
        "input",
        Some(&[Dimension::Fixed(2), Dimension::Fixed(3)]),
        Some(DataType::Float),
    );
    let output = graph.add_value("output", None, None);
    graph.add_input(input);
    graph.add_output(output);
    graph.add_operator(
        "matmul",
        OpType::MatMul,
        &[Some(input), Some(weights)],
        &[output],
    );
    let graph = graph.finish();
    builder.set_graph(graph);
    let mut model = ModelOptions::with_all_ops()
        .enable_optimization(false)
        .load(builder.finish())
        .unwrap();
    let pool = Arc::new(ThreadPool::with_num_threads(1).expect("native thread pool"));
    let options = RunOptions::default().with_thread_pool(Some(Arc::clone(&pool)));
    let input = Tensor::from([[1_f32, 2., 3.], [4., 5., 6.]]);
    let baseline: Tensor<f32> = model
        .run_one(input.clone().into(), Some(options.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        baseline.to_vec(),
        vec![38., 44., 50., 56., 83., 98., 113., 128.]
    );
    model.prepack_weights(&pool);
    let packed: Tensor<f32> = model
        .run_one(input.into(), Some(options.clone()))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(packed, baseline);
    model.prepack_weights(&pool);
    let changed: Tensor<f32> = model
        .run_one(
            Tensor::from([[2_f32, 4., 6.], [8., 10., 12.]]).into(),
            Some(options),
        )
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(
        changed.to_vec(),
        vec![76., 88., 100., 112., 166., 196., 226., 256.]
    );
}

#[test]
fn current_builder_produces_an_executable_graph() {
    let mut builder = ModelBuilder::new();
    let mut graph = builder.graph_builder();
    let constant = graph.add_constant(Tensor::from([1.25f32, -2.5]).view());
    let input = graph.add_value("input", Some(&[Dimension::Fixed(2)]), Some(DataType::Float));
    let added = graph.add_value("added", None, None);
    let output = graph.add_value("output", None, None);
    graph.add_input(input);
    graph.add_output(output);
    graph.add_operator("add", OpType::Add, &[Some(constant), Some(input)], &[added]);
    graph.add_operator("relu", OpType::Relu, &[Some(added)], &[output]);
    let graph = graph.finish();
    builder.set_graph(graph);
    builder.add_metadata(MetadataArgs {
        onnx_hash: Some("builder-custody".into()),
    });
    let model = Model::load(builder.finish()).unwrap();
    assert_eq!(model.metadata().onnx_hash(), Some("builder-custody"));
    let output: Tensor<f32> = model
        .run_one(Tensor::from([3f32, -4.]).into(), None)
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(output.to_vec(), vec![4.25, 0.]);
}

fn model_bytes() -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let shape = builder.create_vector(&[2u32]);
    let constant = sg::ConstantNode::create(
        &mut builder,
        &sg::ConstantNodeArgs {
            shape: Some(shape),
            data_type: sg::ConstantData::NONE,
            data: None,
            data_offset: Some(0),
            dtype: Some(sg::ConstantDataType::Float32),
        },
    );
    let name = builder.create_string("output");
    let node = sg::Node::create(
        &mut builder,
        &sg::NodeArgs {
            name: Some(name),
            data_type: sg::NodeKind::ConstantNode,
            data: Some(constant.as_union_value()),
        },
    );
    let nodes = builder.create_vector(&[node]);
    let outputs = builder.create_vector(&[0u32]);
    let graph = sg::Graph::create(
        &mut builder,
        &sg::GraphArgs {
            nodes: Some(nodes),
            outputs: Some(outputs),
            ..Default::default()
        },
    );
    let description = builder.create_string("current model");
    let metadata = sg::Metadata::create(
        &mut builder,
        &sg::MetadataArgs {
            description: Some(description),
            ..Default::default()
        },
    );
    let root = rten_format::create_model(&mut builder, graph, Some(metadata));
    builder.finish(root, None);
    let graph_bytes = builder.finished_data();
    let header = rten_format::Header {
        model_offset: rten_format::Header::LEN as u64,
        model_len: graph_bytes.len() as u64,
        tensor_data_offset: (rten_format::Header::LEN + graph_bytes.len()) as u64,
    };
    let mut bytes = header.to_buf();
    bytes.extend(graph_bytes);
    for value in [1.25f32, -2.5] {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn run_model(model: Model) {
    assert_eq!(model.metadata().description(), Some("current model"));
    let output = model.node_id("output").unwrap();
    let [value] = model.run_n(vec![], [output], None).unwrap();
    let tensor: Tensor<f32> = value.try_into().unwrap();
    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(tensor.to_vec(), vec![1.25, -2.5]);
}

#[test]
fn current_model_runs_from_bytes_and_disk() {
    let bytes = model_bytes();
    let header = rten_format::Header::from_buf(&bytes).unwrap();
    assert_eq!(header.model_offset, 28);
    let root = rten_format::root_as_model(
        &bytes[header.model_offset as usize..header.tensor_data_offset as usize],
    )
    .unwrap();
    assert_eq!(root.graph().nodes().unwrap().len(), 1);
    assert_eq!(
        root.metadata().unwrap().description(),
        Some("current model")
    );
    for optimize in [false, true] {
        run_model(
            ModelOptions::with_all_ops()
                .enable_optimization(optimize)
                .shape_inference(ShapeInferenceMode::Strict)
                .load(bytes.clone())
                .unwrap(),
        );
    }
    let path = std::env::temp_dir().join(format!(
        "fogscrib-rten-{}-{}.rten",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed),
    ));
    fs::write(&path, &bytes).unwrap();
    run_model(Model::load_file(&path).unwrap());
    #[cfg(feature = "mmap")]
    {
        // SAFETY: This test owns the unchanged file until the model has been dropped.
        run_model(unsafe { Model::load_mmap(&path) }.unwrap());
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn malformed_framing_and_tensor_ranges_fail_closed() {
    let bytes = model_bytes();
    let header = rten_format::Header::from_buf(&bytes).unwrap();
    let mut bad_magic = bytes.clone();
    bad_magic[..4].copy_from_slice(b"NOPE");
    let mut oversized_model = bytes.clone();
    oversized_model[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut overlapping_tensor_data = bytes.clone();
    overlapping_tensor_data[20..28].copy_from_slice(&header.model_offset.to_le_bytes());
    let mut truncated_tensor = bytes.clone();
    truncated_tensor.pop();
    let mut invalid_graph = bytes.clone();
    invalid_graph[28..32].copy_from_slice(&u32::MAX.to_le_bytes());
    let headerless =
        bytes[header.model_offset as usize..header.tensor_data_offset as usize].to_vec();

    for (data, kind) in [
        (bad_magic, LoadErrorKind::UnknownFileType),
        (bytes[..27].to_vec(), LoadErrorKind::ParseError),
        (oversized_model, LoadErrorKind::ParseError),
        (overlapping_tensor_data, LoadErrorKind::ParseError),
        (truncated_tensor, LoadErrorKind::GraphError),
        (invalid_graph, LoadErrorKind::ParseError),
        (headerless, LoadErrorKind::UnknownFileType),
    ] {
        let error = Model::load(data).expect_err("invalid model must be rejected");
        assert_eq!(error.kind(), kind, "{error}");
    }
}

fn attribute_model(kind: sg::OperatorType, with_attributes: bool) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let mut nodes = Vec::new();
    for (shape, dtype, offset) in [
        (vec![2u32], sg::ConstantDataType::Float32, 0),
        (
            if kind == sg::OperatorType::CumSum {
                vec![]
            } else {
                vec![2u32]
            },
            sg::ConstantDataType::Int32,
            8,
        ),
    ] {
        let shape = builder.create_vector(&shape);
        let constant = sg::ConstantNode::create(
            &mut builder,
            &sg::ConstantNodeArgs {
                shape: Some(shape),
                dtype: Some(dtype),
                data_offset: Some(offset),
                ..Default::default()
            },
        );
        nodes.push(sg::Node::create(
            &mut builder,
            &sg::NodeArgs {
                data_type: sg::NodeKind::ConstantNode,
                data: Some(constant.as_union_value()),
                ..Default::default()
            },
        ));
    }
    let value = sg::ValueNode::create(&mut builder, &sg::ValueNodeArgs::default());
    let name = builder.create_string("output");
    nodes.push(sg::Node::create(
        &mut builder,
        &sg::NodeArgs {
            name: Some(name),
            data_type: sg::NodeKind::ValueNode,
            data: Some(value.as_union_value()),
        },
    ));
    let (attrs_type, attrs) = if !with_attributes {
        (sg::OperatorAttrs::NONE, None)
    } else if kind == sg::OperatorType::Shape {
        (
            sg::OperatorAttrs::ShapeAttrs,
            Some(
                sg::ShapeAttrs::create(&mut builder, &sg::ShapeAttrsArgs::default())
                    .as_union_value(),
            ),
        )
    } else if kind == sg::OperatorType::CumSum {
        (
            sg::OperatorAttrs::CumSumAttrs,
            Some(
                sg::CumSumAttrs::create(&mut builder, &sg::CumSumAttrsArgs::default())
                    .as_union_value(),
            ),
        )
    } else {
        assert_eq!(kind, sg::OperatorType::Pad);
        (
            sg::OperatorAttrs::PadAttrs,
            Some(sg::PadAttrs::create(&mut builder, &sg::PadAttrsArgs::default()).as_union_value()),
        )
    };
    let inputs = builder.create_vector(if kind == sg::OperatorType::Shape {
        &[0i32][..]
    } else {
        &[0i32, 1][..]
    });
    let outputs = builder.create_vector(&[2i32]);
    let operator = sg::OperatorNode::create(
        &mut builder,
        &sg::OperatorNodeArgs {
            type_: kind,
            attrs_type,
            attrs,
            inputs: Some(inputs),
            outputs: Some(outputs),
        },
    );
    nodes.push(sg::Node::create(
        &mut builder,
        &sg::NodeArgs {
            data_type: sg::NodeKind::OperatorNode,
            data: Some(operator.as_union_value()),
            ..Default::default()
        },
    ));
    let nodes = builder.create_vector(&nodes);
    let outputs = builder.create_vector(&[2u32]);
    let graph = sg::Graph::create(
        &mut builder,
        &sg::GraphArgs {
            nodes: Some(nodes),
            outputs: Some(outputs),
            ..Default::default()
        },
    );
    let root = rten_format::create_model(&mut builder, graph, None);
    builder.finish(root, None);
    let data = builder.finished_data();
    let mut bytes = rten_format::Header {
        model_offset: rten_format::Header::LEN as u64,
        model_len: data.len() as u64,
        tensor_data_offset: (rten_format::Header::LEN + data.len()) as u64,
    }
    .to_buf();
    bytes.extend(data);
    bytes.extend(1.25f32.to_le_bytes());
    bytes.extend((-2.5f32).to_le_bytes());
    bytes.extend([0u8; 8]);
    bytes
}

#[test]
fn operator_records_are_required_and_current_records_execute() {
    for kind in [
        sg::OperatorType::Shape,
        sg::OperatorType::Pad,
        sg::OperatorType::CumSum,
    ] {
        let error = ModelOptions::with_all_ops()
            .enable_optimization(false)
            .load(attribute_model(kind, false))
            .expect_err("missing current attributes must fail");
        assert_eq!(error.kind(), LoadErrorKind::OperatorInvalid);
        assert!(
            error.to_string().contains("attributes are missing"),
            "{error}"
        );
        let model = Model::load(attribute_model(kind, true)).unwrap();
        let [value] = model
            .run_n(vec![], [model.node_id("output").unwrap()], None)
            .unwrap();
        if kind == sg::OperatorType::Shape {
            let tensor: Tensor<i32> = value.try_into().unwrap();
            assert_eq!(tensor.to_vec(), vec![2]);
        } else {
            let tensor: Tensor<f32> = value.try_into().unwrap();
            let expected = if kind == sg::OperatorType::CumSum {
                vec![1.25, -1.25]
            } else {
                vec![1.25, -2.5]
            };
            assert_eq!(tensor.to_vec(), expected);
        }
    }
}
