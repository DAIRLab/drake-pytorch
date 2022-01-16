workspace(name = "drake-python")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

# Choose which nightly build of Drake to use.
DRAKE_RELEASE = "20220115"  # Can also use "latest"
DRAKE_CHECKSUM = "63bb8a7cbd723cb3d4555dffd4d3606202d24fc92d02eac120f6930636a75512"

http_archive(
    name = "drake_artifacts",
    url = "https://drake-packages.csail.mit.edu/drake/nightly/drake-{}-focal.tar.gz".format(DRAKE_RELEASE),
    sha256 = DRAKE_CHECKSUM,
    strip_prefix = "drake/",
    build_file_content = "#",
)


load("@drake_artifacts//:share/drake/repo.bzl", "drake_repository")

drake_repository(name = "drake")