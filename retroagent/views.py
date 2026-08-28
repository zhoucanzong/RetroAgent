"""Deterministic text, HTML, SVG, and molecule views for research resources."""

from __future__ import annotations

import html
import io
import json
from pathlib import Path
from typing import Any

from retroagent.project import ResearchProject, state_to_dict
from retroagent.resources import ResourceRef, resolve_resource


VIEW_FORMATS = {"json", "text", "html", "svg", "png"}


def render_view(
    project: ResearchProject,
    resource: str,
    *,
    output_format: str = "text",
    output: Path | None = None,
) -> dict[str, Any]:
    output_format = output_format.lower()
    if output_format not in VIEW_FORMATS:
        raise ValueError(f"Unknown view format: {output_format}")
    ledger = project.ledger()
    state = ledger.replay()
    if resource in {"state", "project"}:
        kind = resource
        data = project.status() if resource == "project" else state_to_dict(state)
    elif resource in {"molecules", "hypotheses", "routes", "reactions", "evidence", "experiments"}:
        kind = f"{resource}-collection"
        data = {"items": list(getattr(state, resource).values()), "count": len(getattr(state, resource))}
    else:
        ref = ResourceRef.parse(resource)
        kind = ref.kind
        data = resolve_resource(ref, state, ledger.events)

    if output_format == "json":
        content: str | bytes = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    elif output_format == "text":
        content = _text_view(kind, data)
    elif output_format == "html":
        content = _html_view(kind, data, state_to_dict(state))
    elif output_format == "svg":
        content = _svg_view(kind, data)
    else:
        content = _molecule_png(kind, data)

    written = None
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            output.write_bytes(content)
        else:
            output.write_text(content, encoding="utf-8")
        written = str(output.resolve())
    return {
        "resource": resource,
        "kind": kind,
        "format": output_format,
        "content": None if written else (content.decode("latin1") if isinstance(content, bytes) else content),
        "output": written,
        "bytes": len(content),
    }


def _text_view(kind: str, data: dict[str, Any]) -> str:
    title = data.get("id") or data.get("name") or kind
    lines = [f"{kind}:{title}"]
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines) + "\n"


def _html_view(kind: str, data: dict[str, Any], state: dict[str, Any]) -> str:
    title = str(data.get("id") or data.get("name") or kind)
    body = ""
    if kind == "route":
        reaction_ids = [str(item) for item in data.get("reaction_ids", [])]
        reactions = {item.get("id"): item for item in state.get("reactions", [])}
        cards = []
        for index, reaction_id in enumerate(reaction_ids, 1):
            reaction = reactions.get(reaction_id, {"id": reaction_id})
            cards.append(
                f'<li><strong>{index}. {html.escape(reaction_id)}</strong>'
                f'<span>{html.escape(str(reaction.get("status", "unknown")))}</span>'
                f'<pre>{html.escape(json.dumps(reaction, ensure_ascii=False, indent=2))}</pre></li>'
            )
        body = f'<ol class="route">{"".join(cards) or "<li>No reactions registered</li>"}</ol>'
    elif kind == "evidence":
        body = _definition_list(data)
    elif kind == "evidence-collection":
        rows = []
        for item in data.get("items", []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('id', '')))}</td>"
                f"<td>{html.escape(str(item.get('direction', '')))}</td>"
                f"<td>{html.escape(', '.join(str(value) for value in item.get('target_ids', [])))}</td>"
                f"<td>{html.escape(str(item.get('summary', '')))}</td>"
                "</tr>"
            )
        body = (
            '<table><thead><tr><th>ID</th><th>Direction</th><th>Targets</th><th>Summary</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
        )
    elif kind == "experiment":
        steps = data.get("steps", [])
        body = _definition_list({key: value for key, value in data.items() if key != "steps"})
        body += "<h2>Steps</h2><ol>" + "".join(
            f"<li>{html.escape(json.dumps(step, ensure_ascii=False))}</li>" for step in steps
        ) + "</ol>"
    elif kind == "experiments-collection":
        items = sorted(data.get("items", []), key=lambda item: str(item.get("created_at", item.get("id", ""))))
        body = '<ol class="timeline">' + "".join(
            f'<li><strong>{html.escape(str(item.get("id", "")))}</strong>'
            f'<span>{html.escape(str(item.get("status", "proposed")))}</span>'
            f'<p>{html.escape(str(item.get("objective", item.get("summary", ""))))}</p></li>'
            for item in items
        ) + "</ol>"
    elif kind == "molecule":
        try:
            body = _molecule_svg(data)
        except ValueError:
            body = _definition_list(data)
    elif kind == "state":
        body = "".join(
            f"<section><h2>{html.escape(key.title())}</h2><pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></section>"
            for key, value in data.items()
        )
    else:
        body = _definition_list(data)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RetroAgent | {html.escape(title)}</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:0;color:#18201d;background:#f7f8f7}}main{{max-width:980px;margin:auto;padding:32px 24px}}
h1{{font-size:26px;margin:0 0 20px}}h2{{font-size:17px}}.meta{{color:#52605a;margin-bottom:24px}}dl,li,section{{background:#fff;border:1px solid #d9dfdc;border-radius:6px;padding:16px;margin:0 0 12px}}
dt{{font-weight:700;margin-top:10px}}dd{{margin:4px 0 10px;overflow-wrap:anywhere}}ol{{padding:0;list-style-position:inside}}li span{{float:right;color:#52605a}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f2;padding:12px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{border:1px solid #d9dfdc;padding:10px;text-align:left;vertical-align:top}}
svg{{max-width:100%;height:auto;background:#fff;border:1px solid #d9dfdc}}
</style></head><body><main><h1>{html.escape(title)}</h1><div class="meta">{html.escape(kind)}</div>{body}</main></body></html>"""


def _definition_list(data: dict[str, Any]) -> str:
    items = []
    for key, value in data.items():
        rendered = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
        items.append(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(rendered)}</dd>")
    return f"<dl>{''.join(items)}</dl>"


def _svg_view(kind: str, data: dict[str, Any]) -> str:
    if kind == "molecule":
        return _molecule_svg(data)
    if kind != "route":
        raise ValueError("SVG views currently support route and molecule resources")
    reaction_ids = [str(item) for item in data.get("reaction_ids", [])]
    count = max(1, len(reaction_ids))
    width = 760
    height = 80 + count * 92
    nodes = []
    for index, reaction_id in enumerate(reaction_ids or ["No reactions registered"]):
        y = 32 + index * 92
        nodes.append(f'<rect x="40" y="{y}" width="680" height="58" rx="6" fill="#ffffff" stroke="#75827c"/>')
        nodes.append(f'<text x="60" y="{y + 35}" font-family="Arial" font-size="16" fill="#18201d">{html.escape(reaction_id)}</text>')
        if index:
            nodes.append(f'<line x1="380" y1="{y - 34}" x2="380" y2="{y}" stroke="#2b6f55" stroke-width="3"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(nodes)}</svg>\n'


def _molecule_svg(data: dict[str, Any]) -> str:
    smiles = str(data.get("smiles") or data.get("canonical_smiles") or "")
    if not smiles:
        raise ValueError("Molecule resource does not contain SMILES")
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError as exc:
        raise ValueError("RDKit is required for molecule rendering") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid molecule SMILES: {smiles}")
    return Draw.MolsToGridImage([molecule], molsPerRow=1, subImgSize=(600, 360), useSVG=True)


def _molecule_png(kind: str, data: dict[str, Any]) -> bytes:
    if kind != "molecule":
        raise ValueError("PNG views currently support molecule resources")
    smiles = str(data.get("smiles") or data.get("canonical_smiles") or "")
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError as exc:
        raise ValueError("RDKit is required for molecule rendering") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid molecule SMILES: {smiles}")
    image = Draw.MolToImage(molecule, size=(600, 360))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
