"""ResearchReportTool: structure discovery hypotheses for review.

The tool performs schema normalization and completeness checks only. It does
not decide whether a chemical route is correct.
"""

import json

from retroagent.schemas import build_research_report


class ResearchReportTool:
    name = "build_research_report"
    description = (
        "Normalize a portfolio of chemical route hypotheses into an auditable "
        "research report. Use this before final discovery-mode submission to "
        "ensure every hypothesis has a claim, evidence level, risk review, and "
        "next experiment. This tool does not judge chemistry; it checks report "
        "structure and evidence/risk completeness."
    )

    def execute(self, parameters: dict) -> str:
        report = build_research_report(
            title=parameters.get("title", ""),
            target_smiles=parameters.get("target_smiles", ""),
            hypotheses=parameters.get("hypotheses", []),
        )
        return json.dumps(report, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        evidence_schema = {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["known", "analog-supported", "tool-supported", "speculative"],
                    "description": "Evidence strength supporting this route hypothesis.",
                },
                "source": {"type": "string", "description": "Tool, paper, URL, or rationale source."},
                "summary": {"type": "string", "description": "Short evidence summary."},
                "url": {"type": "string", "description": "Optional source URL."},
                "score": {"type": "number", "description": "Optional numeric confidence or tool score."},
                "metadata": {"type": "object", "description": "Optional extra evidence metadata."},
            },
            "required": ["level", "source", "summary"],
        }
        risk_schema = {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Risk severity.",
                },
                "category": {
                    "type": "string",
                    "description": "Risk category such as stereochemistry, selectivity, safety, IP, or conditions.",
                },
                "message": {"type": "string", "description": "Specific risk or uncertainty."},
                "evidence": {"type": "array", "items": evidence_schema},
            },
            "required": ["severity", "category", "message"],
        }
        audit_schema = {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Audit issue severity.",
                },
                "category": {"type": "string", "description": "Audit category."},
                "message": {"type": "string", "description": "Audit finding."},
                "suggested_fix": {"type": "string", "description": "How to resolve or test the issue."},
                "evidence": {"type": "array", "items": evidence_schema},
            },
            "required": ["severity", "category", "message"],
        }
        experiment_schema = {
            "type": "object",
            "properties": {
                "objective": {"type": "string", "description": "What this experiment tests."},
                "procedure": {"type": "string", "description": "High-level experimental plan."},
                "expected_observation": {
                    "type": "string",
                    "description": "Observation that would support the hypothesis.",
                },
                "decision_point": {
                    "type": "string",
                    "description": "What to do depending on success/failure.",
                },
                "safety_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["objective", "procedure"],
        }
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Optional report title."},
                "target_smiles": {"type": "string", "description": "Target molecule SMILES."},
                "hypotheses": {
                    "type": "array",
                    "description": "Route hypotheses to normalize and check.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "claim": {"type": "string", "description": "Route hypothesis claim."},
                            "target_smiles": {"type": "string"},
                            "route_summary": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["proposed", "tool-supported", "needs-validation", "rejected"],
                            },
                            "evidence": {"type": "array", "items": evidence_schema},
                            "risk_flags": {"type": "array", "items": risk_schema},
                            "audit_issues": {"type": "array", "items": audit_schema},
                            "next_experiments": {
                                "type": "array",
                                "items": experiment_schema,
                            },
                            "metadata": {"type": "object"},
                        },
                        "required": ["claim", "evidence", "next_experiments"],
                    },
                },
            },
            "required": ["target_smiles", "hypotheses"],
        }
