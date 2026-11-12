# Base image for the SAM 3 function (built by deploy.sh, like CVAT's serverless/deploy_cpu.sh).
# No weights here: they are mounted from autolabel/weights at run time.
FROM python:3.12-slim
# short timeout + retries: a stalled download on a slow link fails and retries instead of hanging
ENV PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=10
RUN apt-get update && apt-get -y install --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
# torch wheel (~190 MB) in its own layer so a later failure does not re-download it; curl resumes
RUN curl -fL --retry 100 --retry-all-errors --speed-limit 1000 --speed-time 60 -C - \
        -o /tmp/torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl \
        https://download-r2.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl
# torch deps from PyPI first: the pytorch.org index serves them very slowly here; numpy<2 for sam3
RUN pip install --no-cache-dir "numpy>=1.26,<2" pillow sympy networkx jinja2 fsspec filelock typing_extensions
RUN pip install --no-cache-dir \
        /tmp/torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && rm /tmp/torch-*.whl
# einops, pycocotools, psutil: imported by sam3 at this commit but not declared; triton: imported by sam3, not in CPU torch
RUN pip install --no-cache-dir "setuptools<81" einops pycocotools psutil triton==3.6.0 \
    "git+https://github.com/facebookresearch/sam3.git@f92a5efc4001ebfe9d050d43e61f6fb10e02d73c"
# shortcut: sam3's fused ViT MLP casts to bfloat16 and geometry_encoders pins memory (CUDA only); CPU stays float32.
RUN cd "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/sam3" \
    && sed -i 's/\.to(torch\.bfloat16)//' perflib/fused.py && sed -i 's/scale\.pin_memory()\.to(/scale.to(/' model/geometry_encoders.py \
    && ! grep -n 'bfloat16' perflib/fused.py && ! grep -n 'pin_memory' model/geometry_encoders.py
# shortcut: sam3 hardcodes device="cuda" when precomputing position caches; force CPU. Drop if upstream fixes it.
RUN cd "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/sam3/model" \
    && sed -i 's/device="cuda"/device="cpu"/' position_encoding.py decoder.py \
    && ! grep -n 'device="cuda"' position_encoding.py decoder.py
