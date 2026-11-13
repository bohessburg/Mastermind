from __future__ import annotations

from typing import Any

from .common import Instance, Offer, as_float, as_int, require_api_key, requests_session


class VastAIProvider:
    base_url = "https://console.vast.ai/api/v0"
    default_mode = "native"
    default_image = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"

    def __init__(self, api_key: str | None = None, session: Any | None = None) -> None:
        self.api_key = require_api_key("VAST_API_KEY", api_key)
        self.session = session or requests_session()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _json(self, response: Any) -> Any:
        response.raise_for_status()
        return response.json()

    def search_offers(self, gpu: str, min_vcpus: int, max_price: float) -> list[Offer]:
        params = {
            "q": (
                f"gpu_name contains {gpu} num_cpus>={min_vcpus} "
                f"dph_total<={max_price} rentable=true verified=true"
            )
        }
        data = self._json(self.session.get(f"{self.base_url}/bundles/", headers=self._headers(), params=params))
        raw_offers = data.get("offers", data.get("machines", data if isinstance(data, list) else []))
        offers: list[Offer] = []
        for item in raw_offers:
            offer = Offer(
                id=str(item.get("id", item.get("ask_contract_id", item.get("machine_id", "")))),
                gpu=str(item.get("gpu_name", item.get("gpu", ""))),
                price_per_hour=as_float(item.get("dph_total", item.get("price_per_hour"))),
                vcpus=as_int(item.get("num_cpus", item.get("cpu_cores", item.get("vcpus")))),
                upload_mbps=as_float(item.get("inet_up", item.get("upload_mbps"))),
                raw=dict(item),
            )
            if offer.id and gpu.lower() in offer.gpu.lower() and offer.vcpus >= min_vcpus and offer.price_per_hour <= max_price:
                offers.append(offer)
        return sorted(offers, key=lambda offer: offer.price_per_hour)

    def create_instance(self, offer: Offer | str, image: str) -> Instance:
        offer_id = offer.id if isinstance(offer, Offer) else str(offer)
        payload = {
            "image": image,
            "disk": 80,
            "env": {},
        }
        data = self._json(
            self.session.post(f"{self.base_url}/asks/{offer_id}/", headers=self._headers(), json=payload)
        )
        instance_id = str(data.get("new_contract", data.get("id", data.get("instance_id", ""))))
        return Instance(id=instance_id, status=str(data.get("status", "creating")), raw=dict(data))

    def instance_status(self, instance_id: str) -> Instance:
        data = self._json(self.session.get(f"{self.base_url}/instances/{instance_id}/", headers=self._headers()))
        item = data.get("instance", data)
        ssh_host = str(item.get("ssh_host", item.get("public_ipaddr", "")))
        ssh_port = as_int(item.get("ssh_port", item.get("ports", {}).get("22/tcp", 22)), 22)
        return Instance(
            id=str(item.get("id", instance_id)),
            status=str(item.get("actual_status", item.get("status", "unknown"))),
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=str(item.get("ssh_user", "root")),
            cost_per_hour=as_float(item.get("dph_total", item.get("price_per_hour"))),
            raw=dict(item),
        )

    def ssh_target(self, instance_id: str) -> str:
        return self.instance_status(instance_id).ssh_target()

    def destroy_instance(self, instance_id: str) -> None:
        self._json(self.session.delete(f"{self.base_url}/instances/{instance_id}/", headers=self._headers()))
