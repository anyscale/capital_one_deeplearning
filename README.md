# Distributed Deep Learning with Ray on Anyscale

A hands-on tutorial that takes an ordinary single-GPU PyTorch training loop and
scales it out with Ray Train and Ray Data, then makes it survive failure.

The workload is **next-transaction prediction**: gpt2 fine-tuned on sequences of
credit-card transactions grouped by cardholder, each transaction rendered as a
line of text. It is plain causal language modeling, but over this corpus the
model learns what normal spending looks like for a card — the pretraining stage
underneath a fraud or credit-risk classifier.

## Notebooks

Run them in order. Each is self-contained and takes a few minutes of compute.

| | Topic | You will learn |
|---|---|---|
| `01_ray_foundations.ipynb` | One GPU to many | The three changes that make a loop distributed: `prepare_model`, `get_dataset_shard`, `ray.train.report` — plus Ray Data streaming and tokenization on a CPU actor pool |
| `02_memory_and_sharding.ipynb` | FSDP and DeepSpeed ZeRO | Why replicated data parallel caps model size at one GPU, and two ways to shard past it — under an unchanged Ray Train surface |
| `03_fault_tolerance_and_observability.ipynb` | Resilience | A checkpoint-aware loop, automatic recovery from a worker killed mid-run, elastic worker ranges, throughput, and profiling |

## Cluster

A CPU head node plus T4 GPU workers. The notebooks are written for **2 GPUs**;
`ScalingConfig(num_workers=...)` is the only thing that changes to scale up.

The head node contributes no schedulable CPU, so Ray Data's tokenizer actors run
on the GPU worker nodes. If the cluster is scaled to zero, the first cell that
pulls data will wait for a node to come up.

## Data and models

Everything is staged from a **public S3 mirror** with unsigned reads — no
credentials, and no Hugging Face Hub traffic at runtime (`HF_HUB_OFFLINE=1` is
set on every worker, so an accidental repo-id load raises instead of silently
hitting the Hub). The first cell of each notebook syncs what it needs into
`/mnt/cluster_storage`, which every node can read.

`/mnt/cluster_storage` is scoped to a running cluster and resets with it, so
that staging step re-runs on a fresh cluster. That is expected.

The transaction corpus is derived from IBM's
[TabFormer](https://github.com/IBM/TabFormer) synthetic credit-card dataset
(Apache-2.0). It is synthetic — no real cardholders, no PII.

## Layout

```
common/
  utils.py                     mixed-precision selection, runtime_env, model builders,
                               the tokenized-dataset helper, cluster pretty-printer
  kill_train_worker.py         schedules a real SIGKILL of a training worker, for the
                               fault-tolerance demo
  build_card_transactions.py   provenance: how the transaction corpus was built.
                               Not imported by anything; run it only to rebuild the data
Dockerfile                     pinned image (Ray 2.55.1, torch 2.9.1, DeepSpeed 0.18.9)
```

The teaching-critical Ray calls live inline in the notebooks so they are visible.
`common/` holds only boilerplate the notebooks would otherwise repeat.

## Note when re-running

Ray Train restores automatically from a run's `storage_path` and `name`. Within
one cluster session, re-running notebook 03 finds the completed run and returns
its final result without retraining — which is the point of the manual-restore
cell, but it also means the kill-and-recover demo only fires once per cluster.
Restart the cluster to see it again.
