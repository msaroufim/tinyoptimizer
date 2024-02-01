## Get weights

```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Llama-2-7b-hf")
```


## To run 

```bash
git clone https://github.com/facebookresearch/llama-recipes
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
cd llama-recipes
torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --model_name /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 --fsdp_cpu_offload
```


This still OOMs on 24GB of VRAM even if CPU offloading :(