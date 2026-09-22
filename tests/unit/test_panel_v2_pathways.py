"""Verified anatomical paths remain distinct from measured activation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

from flybrain_interface.panel.app import create_app
from flybrain_interface.panel.pathways_v2 import (
    PATHWAY_ARTIFACT,
    pathway_catalog,
    target_paths,
)


def test_catalog_reports_only_actual_artifact_targets() -> None:
    if not PATHWAY_ARTIFACT.is_file():
        pytest.skip("validated artifact is not installed")
    catalog = pathway_catalog()
    assert catalog["schema_version"] == 2
    assert catalog["available"] is True
    assert len(catalog["targets"]) == 10
    assert catalog["telemetry_available"] is False
    assert catalog["causal_contribution_available"] is False
    assert {row["target_instance"].split("_")[0] for row in catalog["targets"]} == {
        "DNa02",
        "DNg13",
        "DNp09",
        "MDN",
    }


def test_target_routes_are_bounded_and_explicitly_anatomical() -> None:
    if not PATHWAY_ARTIFACT.is_file():
        pytest.skip("validated artifact is not installed")
    target = pathway_catalog()["targets"][0]
    result = target_paths(target["target_neuron_index"])
    assert result is not None
    assert result["source"] == "verified_connectome_anatomy"
    assert result["active_pathway_measurement"] is None
    assert result["causal_attribution"] is None
    assert 1 <= len(result["paths"]) <= 12
    for entry in result["paths"]:
        assert len(entry["synapse_counts"]) == len(entry["neurons"]) - 1
        assert entry["edge_activation"] is None
        assert entry["neurons"][-1]["neuron_index"] == target["target_neuron_index"]
        for neuron in entry["neurons"]:
            assert neuron["measured_voltage_mv"] is None
            assert neuron["measured_drive_mv"] is None
            assert neuron["measured_spike_count"] is None
    assert target_paths(166699) is None


def test_target_http_routes_validate_requested_indices(tmp_path: Path) -> None:
    app = create_app(tmp_path, tmp_path / "runs")
    routes = {
        route.path: route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute)
    }
    if PATHWAY_ARTIFACT.is_file():
        result = asyncio.run(routes["/api/v2/pathways"]())
        assert result["available"] is True
        selected = result["targets"][0]["target_neuron_index"]
        mapped = asyncio.run(routes["/api/v2/pathways/{target_index}"](selected))
        assert mapped["target_neuron_index"] == selected
    with pytest.raises(HTTPException) as missing:
        asyncio.run(routes["/api/v2/pathways/{target_index}"](-1))
    assert missing.value.status_code == 404


def test_pathway_observation_resolves_real_indices_and_rejects_unknowns() -> None:
    from flybrain_interface.panel.models import PathwayObservationConfig
    from flybrain_interface.panel.pathway_observation import resolve_observation

    target = pathway_catalog()["targets"][0]
    observation = PathwayObservationConfig(
        target_neuron_index=target["target_neuron_index"], path_index=0
    )
    selected = resolve_observation(observation)
    route = target_paths(target["target_neuron_index"])["paths"][0]
    assert selected["source"] == "verified_connectome_anatomy"
    assert selected["hop_count"] in (2, 3)
    assert selected["neuron_indices"] == [
        row["neuron_index"] for row in route["neurons"]
    ]
    assert len(selected["neuron_indices"]) <= 4
    assert selected["neuron_indices"][-1] == target["target_neuron_index"]
    with pytest.raises(ValueError, match="unknown anatomical pathway"):
        resolve_observation(
            PathwayObservationConfig(
                target_neuron_index=target["target_neuron_index"], path_index=11
            )
            if len(target_paths(target["target_neuron_index"])["paths"]) < 12
            else PathwayObservationConfig(target_neuron_index=166699, path_index=0)
        )


def test_observation_route_only_accepts_server_verified_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flybrain_interface.panel.models import (
        PathwayObservationConfig,
        PathwayObserveRequest,
    )

    app = create_app(tmp_path, tmp_path / "runs")
    endpoint = {
        item.path: item.endpoint for item in app.routes if isinstance(item, APIRoute)
    }["/api/v2/pathways/observe"]
    target = pathway_catalog()["targets"][0]
    seen: list[dict[str, object] | None] = []
    monkeypatch.setattr(app.state.controller, "observe_pathway", seen.append)
    request = PathwayObserveRequest(
        observation=PathwayObservationConfig(
            target_neuron_index=target["target_neuron_index"], path_index=0
        )
    )
    with pytest.raises(HTTPException) as inactive:
        asyncio.run(endpoint(request))
    assert inactive.value.status_code == 409
    app.state.controller.latest = {"status": "running"}
    accepted = asyncio.run(endpoint(request))
    assert accepted.accepted
    assert len(seen[0]["neuron_indices"]) <= 4
    assert seen[0]["target_neuron_index"] == target["target_neuron_index"]
    cleared = asyncio.run(endpoint(PathwayObserveRequest(observation=None)))
    assert cleared.accepted
    assert seen[-1] is None
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(
            endpoint(
                PathwayObserveRequest(
                    observation=PathwayObservationConfig(
                        target_neuron_index=166699, path_index=0
                    )
                )
            )
        )
    assert invalid.value.status_code == 422
    with pytest.raises(ValueError):
        PathwayObserveRequest.model_validate(
            {
                "observation": {
                    "target_neuron_index": target["target_neuron_index"],
                    "path_index": 0,
                    "neuron_indices": [0, 1, 2],
                }
            }
        )
