# CrediBench Client

Python client for the CrediGraph API.

## Install

```bash
pip install credigraph
```

## Usage

```python
from credigraph import query
print(query("apnews.com"))
print(query(["apnews.com", "cnn.com"]))
```

With auth: `export HF_TOKEN=hf_...` 

Or, 
```python
from credigraph import CrediGraphClient
c = CrediGraphClient(token="hf_...")
c.query("reuters.com")
```