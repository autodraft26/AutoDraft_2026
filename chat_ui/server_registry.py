import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _norm_model_id(model_id: str) -> str:
    return str(model_id or "").strip().lower()


def _dedupe_models(models: List["ModelSpec"]) -> List["ModelSpec"]:
    out = []
    seen = set()
    for m in models:
        key = _norm_model_id(m.model_id)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


@dataclass
class ModelSpec:
    model_id: str
    label: str
    pricing_input_per_1m: Optional[float] = None
    pricing_output_per_1m: Optional[float] = None
    supports_tools: bool = False

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "pricing_input_per_1m": self.pricing_input_per_1m,
            "pricing_output_per_1m": self.pricing_output_per_1m,
            "supports_tools": self.supports_tools,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelSpec":
        return cls(
            model_id=str(data.get("model_id", "")),
            label=str(data.get("label", data.get("model_id", ""))),
            pricing_input_per_1m=(
                float(data["pricing_input_per_1m"])
                if data.get("pricing_input_per_1m") is not None
                else None
            ),
            pricing_output_per_1m=(
                float(data["pricing_output_per_1m"])
                if data.get("pricing_output_per_1m") is not None
                else None
            ),
            supports_tools=bool(data.get("supports_tools", False)),
        )


@dataclass
class ServerSpec:
    server_id: str
    name: str
    source: str  # catalog | user
    server_type: str  # external | local
    protocol: str  # openai_chat_completions | autodraft_target
    endpoint: str
    requires_api_key: bool = False
    api_key: Optional[str] = None
    api_key_hint: str = ""
    enabled: bool = True
    default_model_id: Optional[str] = None
    models: List[ModelSpec] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self, include_secret: bool = False) -> Dict:
        out = {
            "server_id": self.server_id,
            "name": self.name,
            "source": self.source,
            "server_type": self.server_type,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "requires_api_key": self.requires_api_key,
            "has_api_key": bool(self.api_key),
            "api_key_hint": self.api_key_hint,
            "enabled": self.enabled,
            "default_model_id": self.default_model_id,
            "models": [m.to_dict() for m in self.models],
            "metadata": dict(self.metadata),
        }
        if include_secret:
            out["api_key"] = self.api_key
        return out

    @classmethod
    def from_dict(cls, data: Dict) -> "ServerSpec":
        models = data.get("models") or []
        parsed_models = [ModelSpec.from_dict(m) for m in models if isinstance(m, dict)]
        parsed_models = _dedupe_models(parsed_models)
        return cls(
            server_id=str(data.get("server_id", _uid("srv"))),
            name=str(data.get("name", "Unnamed server")),
            source=str(data.get("source", "user")),
            server_type=str(data.get("server_type", "external")),
            protocol=str(data.get("protocol", "openai_chat_completions")),
            endpoint=str(data.get("endpoint", "")).rstrip("/"),
            requires_api_key=bool(data.get("requires_api_key", False)),
            api_key=(str(data["api_key"]) if data.get("api_key") else None),
            api_key_hint=str(data.get("api_key_hint", "")),
            enabled=bool(data.get("enabled", True)),
            default_model_id=(
                str(data["default_model_id"]) if data.get("default_model_id") else None
            ),
            models=parsed_models,
            metadata=dict(data.get("metadata") or {}),
        )


def default_catalog_servers() -> List[ServerSpec]:
    # Start with an empty catalog; users add servers manually via UI.
    return []


class ServerRegistry:
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._servers: Dict[str, ServerSpec] = {}
        self._hidden_catalog_ids = set()
        self._load()

    def _load(self):
        self._servers = {s.server_id: s for s in default_catalog_servers()}
        self._hidden_catalog_ids = set()
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            hidden_ids = data.get("hidden_catalog_ids", [])
            if isinstance(hidden_ids, list):
                self._hidden_catalog_ids = {str(x) for x in hidden_ids if str(x).strip()}
            for row in data.get("servers", []):
                if not isinstance(row, dict):
                    continue
                spec = ServerSpec.from_dict(row)
                self._servers[spec.server_id] = spec
            for sid in list(self._hidden_catalog_ids):
                spec = self._servers.get(sid)
                if spec and spec.source == "catalog":
                    self._servers.pop(sid, None)
        except Exception:
            return

    def save(self):
        rows = [
            spec.to_dict(include_secret=True)
            for spec in self._servers.values()
            if spec.source == "user"
        ]
        payload = {
            "servers": rows,
            "hidden_catalog_ids": sorted(self._hidden_catalog_ids),
        }
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def list_servers(self) -> List[Dict]:
        rows = [spec.to_dict(include_secret=False) for spec in self._servers.values()]
        rows.sort(key=lambda r: (r.get("source") != "catalog", r.get("name", "")))
        return rows

    def get(self, server_id: str) -> Optional[ServerSpec]:
        return self._servers.get(server_id)

    def add_user_server(
        self,
        name: str,
        endpoint: str,
        protocol: str,
        server_type: str,
        requires_api_key: bool,
        models: List[Dict],
        default_model_id: Optional[str],
        api_key: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ServerSpec:
        parsed_models = []
        for m in models:
            if not isinstance(m, dict):
                continue
            model_id = str(m.get("model_id", "")).strip()
            if not model_id:
                continue
            parsed_models.append(
                ModelSpec(
                    model_id=model_id,
                    label=str(m.get("label", model_id)),
                    pricing_input_per_1m=(
                        float(m["pricing_input_per_1m"])
                        if m.get("pricing_input_per_1m") is not None
                        else None
                    ),
                    pricing_output_per_1m=(
                        float(m["pricing_output_per_1m"])
                        if m.get("pricing_output_per_1m") is not None
                        else None
                    ),
                )
            )
        parsed_models = _dedupe_models(parsed_models)
        server_id = _uid("user")
        spec = ServerSpec(
            server_id=server_id,
            name=name.strip() or "User Server",
            source="user",
            server_type=server_type if server_type in {"external", "local"} else "external",
            protocol=protocol or "openai_chat_completions",
            endpoint=endpoint.strip().rstrip("/"),
            requires_api_key=bool(requires_api_key),
            api_key=api_key.strip() if isinstance(api_key, str) and api_key.strip() else None,
            default_model_id=default_model_id,
            models=parsed_models,
            metadata=dict(metadata or {}),
        )
        self._servers[spec.server_id] = spec
        self.save()
        return spec

    def set_api_key(self, server_id: str, api_key: str) -> bool:
        spec = self._servers.get(server_id)
        if not spec:
            return False
        spec.api_key = api_key.strip() if api_key else None
        self.save()
        return True

    def set_enabled(self, server_id: str, enabled: bool) -> bool:
        spec = self._servers.get(server_id)
        if not spec:
            return False
        spec.enabled = bool(enabled)
        self.save()
        return True

    def remove_user_server(self, server_id: str) -> bool:
        spec = self._servers.get(server_id)
        if not spec or spec.source != "user":
            return False
        self._servers.pop(server_id, None)
        self.save()
        return True

    def remove_server(self, server_id: str) -> bool:
        spec = self._servers.get(server_id)
        if not spec:
            return False
        if spec.source == "catalog":
            self._hidden_catalog_ids.add(str(server_id))
        self._servers.pop(server_id, None)
        self.save()
        return True
