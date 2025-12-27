from .client import CrediGraphClient

def query(domain: str, **kwargs):
    return CrediGraphClient(**kwargs).query(domain)