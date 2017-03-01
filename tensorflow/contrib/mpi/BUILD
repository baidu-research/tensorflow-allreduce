# Ops that communicate with other processes via MPI.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

tf_custom_op_library(
    name = "mpi.so",
    srcs = ["mpi_ops.cc", "ring.cc", "ring.h"],
    gpu_srcs = ["ring.cu.cc", "ring.h"],
    deps = [
        "//third_party/mpi:mpi",
        ":mpi_message_proto_cc",
    ],
)

tf_py_test(
    name = "mpi_ops_test",
    srcs = ["mpi_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
    data = [
        ":mpi.so",
    ],
    tags = ["manual"],
)

py_library(
    name = "mpi_ops_py",
    srcs = [
        "__init__.py",
        "mpi_ops.py",
    ],
    data = [
        ":mpi.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_proto_library(
    name = "mpi_message_proto",
    srcs = ["mpi_message.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
