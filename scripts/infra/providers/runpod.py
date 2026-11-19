from __future__ import annotations

from typing import Any

from .common import Instance, Offer, as_float, as_int, require_api_key, requests_session


class RunPodProvider:
    base_url = "https://api.runpod.io/graphql"
    default_mode = "docker"
    default_image = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"

    def __init__(self, api_key: str | None = None, session: Any | None = None) -> None:
        self.api_key = require_api_key("RUNPOD_API_KEY", api_key)
        self.session = session or requests_session()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.post(
            self.base_url,
            headers=self._headers(),
            json={"query": query, "variables": variables or {}},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("errors"):
            raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
        return dict(data.get("data", {}))

    def search_offers(self, gpu: str, min_vcpus: int, max_price: float) -> list[Offer]:
        query = """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            lowestPrice(input: {gpuCount: 1})
          }
        }
        """
        data = self._graphql(query)
        offers: list[Offer] = []
        for item in data.get("gpuTypes", []):
            name = str(item.get("displayName", item.get("id", "")))
            price = as_float(item.get("lowestPrice"))
            # RunPod exposes CPU choices at pod creation time; keep the requested
            # vCPU floor in the offer so the CLI prints the target shape.
            offer = Offer(
                id=str(item.get("id", name)),
                gpu=name,
                price_per_hour=price,
                vcpus=min_vcpus,
                upload_mbps=0.0,
                raw=dict(item),
            )
            if gpu.lower() in offer.gpu.lower() and offer.price_per_hour <= max_price:
                offers.append(offer)
        return sorted(offers, key=lambda offer: offer.price_per_hour)

    def create_instance(self, offer: Offer | str, image: str) -> Instance:
        gpu_type = offer.id if isinstance(offer, Offer) else str(offer)
        mutation = """
        mutation CreatePod($input: PodRentInterruptableInput!) {
          podRentInterruptable(input: $input) { id desiredStatus }
        }
        """
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuTypeId": gpu_type,
                "gpuCount": 1,
                "containerDiskInGb": 80,
                "volumeInGb": 100,
                "imageName": image,
                "dockerArgs": "",
                "ports": "22/tcp",
            }
        }
        data = self._graphql(mutation, variables)
        pod = data.get("podRentInterruptable", {})
        return Instance(id=str(pod.get("id", "")), status=str(pod.get("desiredStatus", "creating")), raw=dict(pod))

    def instance_status(self, instance_id: str) -> Instance:
        query = """
        query Pod($id: String!) {
          pod(input: {podId: $id}) {
            id
            desiredStatus
            runtime { ports { ip isIpPublic privatePort publicPort type } }
            costPerHr
          }
        }
        """
        pod = self._graphql(query, {"id": instance_id}).get("pod", {})
        host = ""
        port = 22
        runtime = pod.get("runtime") or {}
        for entry in runtime.get("ports", []) or []:
            if as_int(entry.get("privatePort")) == 22 and entry.get("isIpPublic"):
                host = str(entry.get("ip", ""))
                port = as_int(entry.get("publicPort"), 22)
                break
        return Instance(
            id=str(pod.get("id", instance_id)),
            status=str(pod.get("desiredStatus", "unknown")),
            ssh_host=host,
            ssh_port=port,
            ssh_user="root",
            cost_per_hour=as_float(pod.get("costPerHr")),
            raw=dict(pod),
        )

    def ssh_target(self, instance_id: str) -> str:
        return self.instance_status(instance_id).ssh_target()

    def destroy_instance(self, instance_id: str) -> None:
        mutation = "mutation StopPod($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }"
        self._graphql(mutation, {"id": instance_id})
