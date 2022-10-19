#!/usr/bin/env bash
# shellcheck shell=bash
set -eux

function download_from_pytorch_wheel() {
    # torch, torchvision
    local package_name="${1}"
    local package_version="${2}"
    local python_version="${3:-3.7}"
    local os_type="${4:-linux}"
    local arch="${5:-x86_64}"
    local cuda_version="${6:-9.0}"

    python_version="$(echo "${3:-3.7}" | tr -d .)"
    cuda_version="$(echo "${6:-9.0}" | tr -d .)"
    local pytorch_wheel_name="${package_name}-${package_version}-cp${python_version}-cp${python_version}m-${os_type}_${arch}.whl"
    local pytorch_wheel_url="https://download.pytorch.org/whl/cu${cuda_version}/${pytorch_wheel_name}"
    local pytorch_wheel_path="${pytorch_wheel_name}"
    if [ ! -f "${pytorch_wheel_path}" ]; then
        wget -O "${pytorch_wheel_path}" "${pytorch_wheel_url}"
    fi
}

(
    mkdir -p .wheel
    cd .wheel
    download_from_pytorch_wheel torch 1.1.0 3.7 linux x86_64
    download_from_pytorch_wheel torchvision 0.3.0 3.7 manylinux1 x86_64
)
