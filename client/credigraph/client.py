import os
import requests
from typing import List

DEFAULT_API_URL = "https://ekmpa-credibench-api.hf.space"

class CrediGraphClient:
    def __init__(self, api_url=None, token=None):
        self.api_url = api_url or DEFAULT_API_URL
        self.token = token or os.getenv("HF_TOKEN")

    def query_domain(self, domain: str):
        url = f"{self.api_url}/by_domain/{domain}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def query(self, domains: List[str] | str):
        if isinstance(domains, str):
            return self.query_domain(domains)

        return [self.query_domain(d) for d in domains]