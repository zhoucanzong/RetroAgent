import json
import io
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
import tempfile
import unittest
from pathlib import Path

from retroagent.tools.retrochimera import (
    DisconnectionEnsembleTool,
    RetroChimeraPredictionTool,
    NeuralLocPredictionTool,
    TransformerPredictionTool,
    ensemble_disconnections,
)
from retroagent.run.retro import _build_environment


TARGET = "CCOC(=O)C"


class RetrosynthesisEnsembleTests(unittest.TestCase):
    def test_default_environment_exposes_both_fusion_tools(self):
        names = {tool["name"] for tool in _build_environment().get_tools_spec()}

        self.assertIn("retrochimera_predict", names)
        self.assertIn("ensemble_disconnections", names)
        self.assertIn("neuralloc_predict", names)
        self.assertIn("transformer_predict", names)

    def test_rank_weighted_consensus_promotes_cross_source_agreement(self):
        result = ensemble_disconnections(
            target_smiles=TARGET,
            sources=[
                {"name": "editing", "predictions": [
                    {"precursor_smiles_list": ["CCCl", "O"], "rank": 1},
                    {"precursor_smiles_list": ["CCO", "CC(=O)O"], "rank": 2},
                ]},
                {"name": "de_novo", "predictions": [
                    {"precursors": ["CC(=O)O", "CCO"], "rank": 1},
                    {"precursors": ["CCBr", "O"], "rank": 2},
                ]},
            ],
            rank_weights={"editing": [1.0, 0.6], "de_novo": [1.0, 0.6]},
        )

        best = result["reactions"][0]
        self.assertEqual(best["precursor_smiles_list"], ["CC(=O)O", "CCO"])
        self.assertEqual(best["source_count"], 2)
        self.assertEqual(best["individual_ranks"], {"editing": 2, "de_novo": 1})
        self.assertGreater(best["ensemble_score"], result["reactions"][1]["ensemble_score"])
        self.assertAlmostEqual(sum(item["probability"] for item in result["reactions"]), 1.0)

    def test_consensus_only_keeps_predictions_shared_by_every_source(self):
        result = ensemble_disconnections(
            target_smiles=TARGET,
            sources=[
                {"name": "template", "predictions": [
                    {"reactants": "CCO.CC(=O)O"}, {"reactants": "CCCl.O"},
                ]},
                {"name": "language", "predictions": [
                    {"reactants": "CC(=O)O.CCO"}, {"reactants": "CCBr.O"},
                ]},
            ],
            consensus_only=True,
        )

        self.assertEqual(result["count"], 1)
        self.assertTrue(result["reactions"][0]["unanimous"])
        self.assertFalse(result["ensemble_summary"]["calibrated_weights"])
        self.assertIn("reciprocal-rank", " ".join(result["warnings"]))

    def test_weight_profile_must_be_monotone_convex(self):
        sources = [
            {"name": "a", "predictions": [{"reactants": "CCO.CC(=O)O"}]},
            {"name": "b", "predictions": [{"reactants": "CCO.CC(=O)O"}]},
        ]
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            ensemble_disconnections(
                target_smiles=TARGET,
                sources=sources,
                rank_weights={"a": [0.5, 1.0], "b": [1.0, 0.5]},
            )
        with self.assertRaisesRegex(ValueError, "convex"):
            ensemble_disconnections(
                target_smiles=TARGET,
                sources=sources,
                rank_weights={"a": [1.0, 0.9, 0.1], "b": [1.0, 0.5, 0.25]},
            )
        with self.assertRaisesRegex(ValueError, "missing sources: b"):
            ensemble_disconnections(
                target_smiles=TARGET,
                sources=sources,
                rank_weights={"a": [1.0, 0.5]},
            )

    def test_tool_contract_is_machine_readable(self):
        tool = DisconnectionEnsembleTool()
        result = json.loads(tool.execute({
            "target_smiles": TARGET,
            "sources": [
                {"name": "a", "predictions": [{"reactants": "CCO.CC(=O)O"}]},
                {"name": "b", "predictions": [{"reactants": "CC(=O)O.CCO"}]},
            ],
        }))

        self.assertEqual(result["count"], 1)
        self.assertIn("required", tool.parameters_schema())

    def test_retrochimera_adapter_preserves_constituent_provenance(self):
        class Molecule:
            def __init__(self, smiles):
                self.smiles = smiles

        class Prediction:
            reactants = [Molecule("CCO"), Molecule("CC(=O)O")]
            metadata = {
                "score": 1.7,
                "probability": 0.81,
                "individual_ranks": {"neural_loc": 1, "r_smiles_2": 0},
                "individual_metadata": {"neural_loc": {"score": 0.7}},
            }

        class Model:
            def __call__(self, inputs, num_results):
                self.inputs = inputs
                self.num_results = num_results
                return [[Prediction()]]

        model = Model()
        with tempfile.TemporaryDirectory() as tmp:
            tool = RetroChimeraPredictionTool(
                model_factory=lambda **kwargs: model,
                molecule_factory=Molecule,
            )
            result = json.loads(tool.execute({
                "target_smiles": TARGET,
                "model_dir": tmp,
                "num_results": 5,
                "consensus_only": True,
            }))

        reaction = result["reactions"][0]
        self.assertEqual(reaction["sources"], ["neural_loc", "r_smiles_2"])
        self.assertEqual(reaction["individual_ranks"]["r_smiles_2"], 0)
        self.assertEqual(reaction["probability"], 0.81)
        self.assertTrue(result["consensus_only"])

    def test_retrochimera_adapter_requires_explicit_checkpoint_directory(self):
        tool = RetroChimeraPredictionTool(model_factory=lambda **kwargs: None, molecule_factory=str)
        with self.assertRaisesRegex(ValueError, "model directory is required"):
            tool.execute({"target_smiles": TARGET})

    def test_submodel_options_are_validated_and_part_of_model_cache_key(self):
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return lambda inputs, num_results: [[]]

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "models.json").write_text(json.dumps({"editing": ["Model", [1]]}))
            tool = RetroChimeraPredictionTool(model_factory=factory, molecule_factory=str)
            params = {"target_smiles": TARGET, "model_dir": tmp,
                      "submodel_kwargs": {"editing": {"num_processes": 2}}}
            tool.execute(params)
            tool.execute(params)
            tool.execute({**params, "submodel_kwargs": {"editing": {"num_processes": 1}}})
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["editing"], {"num_processes": 2})
            with self.assertRaisesRegex(ValueError, "absent from checkpoint"):
                tool.execute({**params, "submodel_kwargs": {"missing": {}}})

    def test_standalone_tools_resolve_checkpoint_component_and_share_bounded_cache(self):
        calls, closed = [], []
        class Model:
            def __init__(self, directory):
                self._server = type("Server", (), {"close": lambda _: closed.append(directory)})()
            def __call__(self, inputs, num_results):
                return [[type("Prediction", (), {
                    "reactants": [type("Molecule", (), {"smiles": "CCO"})()],
                    "metadata": {"probability": 0.9},
                })()]]
        def factory(**kwargs):
            calls.append(kwargs)
            return Model(kwargs["model_dir"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for name in ("edit", "smiles"):
                (root / name).mkdir()
            (root / "models.json").write_text(json.dumps({
                "edit": ["TemplateLocalizationModel", [1]],
                "smiles": ["SmilesTransformerModel", [1]],
            }))
            cache = {}
            edit = NeuralLocPredictionTool(model_factory=factory, molecule_factory=str, model_cache=cache)
            transformer = TransformerPredictionTool(model_factory=factory, molecule_factory=str, model_cache=cache)
            args = {"target_smiles": TARGET, "model_dir": tmp, "model_kwargs": {"num_processes": 2}}
            result = json.loads(edit.execute(args))
            self.assertEqual(calls[-1]["model_dir"], root / "edit")
            self.assertEqual(calls[-1]["num_processes"], 2)
            self.assertNotIn("consensus_only", calls[-1])
            self.assertEqual(result["ensemble_source"]["source_ids"], [str(root / "edit")])
            direct = json.loads(edit.execute({**args, "model_dir": str(root / "edit")}))
            default_options = json.loads(edit.execute({**args, "model_kwargs": {}}))
            self.assertEqual(default_options["inference_options"]["model_options"]["num_processes"], 2)
            self.assertEqual(result["reactions"][0]["id"], direct["reactions"][0]["id"])
            self.assertEqual(len(calls), 1)
            transformer.execute({"target_smiles": TARGET, "model_dir": tmp})
            self.assertEqual(closed, [root / "edit"])
            self.assertEqual(len(cache), 1)
            transformer.close()
            self.assertEqual(cache, {})

    def test_fusion_rejects_nested_or_aliased_duplicate_provenance(self):
        prediction = {"reactants": "CCO", "source_ids": ["weights/edit"]}
        for first in (
            {"name": "ensemble", "source_ids": ["weights/edit", "weights/smiles"], "predictions": [prediction]},
            {"name": "renamed_edit", "predictions": [prediction]},
        ):
            with self.assertRaisesRegex(ValueError, "Overlapping source_ids"):
                ensemble_disconnections(target_smiles=TARGET, sources=[
                    first, {"name": "edit", "predictions": [prediction]},
                ])

    def test_reaction_ids_do_not_collide_across_targets_and_mapping_is_ignored(self):
        sources = [
            {"name": "a", "predictions": [{"reactants": "[CH3:1][CH2:2][OH:3]"}]},
            {"name": "b", "predictions": [{"reactants": "OCC"}]},
        ]
        first = ensemble_disconnections(target_smiles=TARGET, sources=sources)
        second = ensemble_disconnections(target_smiles="CC=O", sources=sources)
        self.assertEqual(first["count"], 1)
        self.assertEqual(first["reactions"][0]["source_count"], 2)
        self.assertNotEqual(first["reactions"][0]["id"], second["reactions"][0]["id"])

    def test_fusion_rejects_wrong_target(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            ensemble_disconnections(target_smiles=TARGET, sources=[
                {"name": "a", "predictions": [{"reactants": "CCO", "product_smiles": "CC=O"}]},
                {"name": "b", "predictions": []},
            ])

    def test_model_diagnostics_do_not_pollute_json_stdout(self):
        def factory(**kwargs):
            print("model startup")
            return lambda inputs, num_results: [[]]
        tool = TransformerPredictionTool(model_factory=factory, molecule_factory=str)
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(stdout), redirect_stderr(stderr):
            result = tool.execute({"target_smiles": TARGET, "model_dir": tmp})
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("model startup", stderr.getvalue())
        self.assertEqual(json.loads(result)["count"], 0)

    def test_spawned_worker_stdout_is_redirected_and_restored(self):
        code = (
            "import subprocess,sys; from retroagent.tools.retrochimera import _diagnostic_output\n"
            "with _diagnostic_output():\n"
            " subprocess.run([sys.executable, '-c', 'print(\"worker diagnostic\")'], check=True)\n"
            "print('{}')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), {})
        self.assertIn("worker diagnostic", result.stderr)


if __name__ == "__main__":
    unittest.main()
