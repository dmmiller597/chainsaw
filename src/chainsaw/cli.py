"""Command-line entry point for running Chainsaw domain predictions.

Accepts a single structure file or a directory of PDB/MMCIF files and writes a
tab-separated table of predicted domain boundaries.
"""

import argparse
import hashlib
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from torch import compile as torch_compile

from chainsaw import constants, featurisers
from chainsaw.domain_assignment.util import convert_domain_dict_strings
from chainsaw.factories import pairwise_predictor
from chainsaw.models.results import PredictionResult
from chainsaw.prediction_result_file import PredictionResultsFile
from chainsaw.utils import common as common_utils
from chainsaw.utils.pymol_3d_visuals import generate_pymol_image

LOG = logging.getLogger(__name__)
ACCEPTED_STRUCTURE_FILE_SUFFIXES = frozenset({".pdb", ".cif", ".mmcif"})


@dataclass(frozen=True, slots=True)
class Segment:
    """A single contiguous residue range within a domain, in 1-based residue indices."""

    domain_id: str
    start_index: int
    end_index: int


@dataclass(slots=True)
class Domain:
    """A domain made up of one or more (possibly discontinuous) segments."""

    domain_id: str
    segments: list[Segment] = field(default_factory=list)


def setup_logging() -> None:
    loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
    # log everything to stderr so results can go to stdout
    logging.basicConfig(
        level=loglevel,
        stream=sys.stderr,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )


def get_input_method(args: argparse.Namespace) -> str:
    """Return the single input method the user selected, erroring if not exactly one."""
    inputs = {
        "uniprot_id": getattr(args, "uniprot_id", None),
        "uniprot_id_list_file": getattr(args, "uniprot_id_list_file", None),
        "structure_directory": getattr(args, "structure_directory", None),
        "structure_file": getattr(args, "structure_file", None),
        "pdb_id_list_file": getattr(args, "pdb_id_list_file", None),
        "pdb_id": getattr(args, "pdb_id", None),
    }
    selected = [name for name, value in inputs.items() if value is not None]
    if len(selected) != 1:
        raise ValueError("Exactly one input method must be provided")
    return selected[0]


def load_model(
    *,
    model_dir: str | Path,
    remove_disordered_domain_threshold: float = 0.35,
    min_ss_components: int = 2,
    min_domain_length: int = 30,
    post_process_domains: bool = True,
    compile_model: bool = False,
    force_cpu: bool = False,
    use_mps_if_available: bool = False,
) -> Any:
    """Load a trained Chainsaw model and configure its post-processing thresholds."""
    model_path = Path(model_dir).expanduser()
    required_files = ("config.json", "feature_config.json", "weights.pt")
    missing = [name for name in required_files if not (model_path / name).is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Model directory '{model_path}' is missing required file(s): {missing_list}")

    config = common_utils.load_json(model_path / "config.json")
    feature_config = common_utils.load_json(model_path / "feature_config.json")
    learner_config = config["learner"]
    data_config = config["data"]
    learner_config["remove_disordered_domain_threshold"] = remove_disordered_domain_threshold
    learner_config["post_process_domains"] = post_process_domains
    learner_config["min_ss_components"] = min_ss_components
    learner_config["min_domain_length"] = min_domain_length
    learner_config["dist_transform_type"] = data_config.get("dist_transform") or "min_replace_inverse"
    # Explicit JSON null / missing must not override the PairwiseDomainPredictor default of 10 —
    # unidoc_exponent divides by this value.
    distance_denominator = data_config.get("distance_denominator")
    learner_config["distance_denominator"] = 10 if distance_denominator is None else distance_denominator
    # Any: the model gains a feature_config attribute and may be wrapped by torch.compile.
    learner: Any = pairwise_predictor(
        learner_config,
        output_dir=str(model_path),
        force_cpu=force_cpu,
        use_mps_if_available=use_mps_if_available,
    )
    learner.feature_config = feature_config
    learner.load_checkpoints()
    learner.eval()
    if compile_model:
        try:
            learner = torch_compile(learner)
        except Exception:
            LOG.warning("torch.compile unavailable, running uncompiled model", exc_info=True)
    return learner


def _residue_label(index: int, labels_by_index: dict[int, str]) -> str:
    if index not in labels_by_index:
        raise ValueError(f"Index {index} not in residue label map ({labels_by_index})")
    return labels_by_index[index]


def predict(
    model: Any,
    pdb_path: str | Path,
    *,
    renumber_pdbs: bool = True,
    pdbchain: str | None = None,
) -> PredictionResult:
    """Run the model on a single structure and return its :class:`PredictionResult`."""
    start = perf_counter()
    structure_path = Path(pdb_path).expanduser()

    model_structure = featurisers.get_model_structure(structure_path)

    if pdbchain is None:
        first_chain = next(model_structure.get_chains(), None)
        if first_chain is None:
            raise ValueError(f"Structure '{structure_path}' does not contain any chains")
        pdbchain = str(first_chain.id)
        LOG.info("No chain specified for %s; using first chain '%s'", structure_path, pdbchain)

    model_residues = featurisers.get_model_structure_residues(model_structure, chain=pdbchain)
    if not model_residues:
        raise ValueError(f"Chain '{pdbchain}' in '{structure_path}' contains no supported amino-acid residues")
    labels_by_index = {int(r.index): str(r.res_label) for r in model_residues}
    model_structure_seq = "".join(r.aa for r in model_residues)
    model_structure_md5 = hashlib.md5(model_structure_seq.encode("utf-8"), usedforsecurity=False).hexdigest()

    x = featurisers.inference_time_create_features(
        structure_path,
        feature_config=model.feature_config,
        chain=pdbchain,
        renumber_pdbs=renumber_pdbs,
        model_structure=model_structure,
    )

    _a_hat, domain_dict, confidence = model.predict(x)
    # convert 0-indexed to 1-indexed to match AlphaFold indexing
    domain_dict = [{k: [r + 1 for r in v] for k, v in d.items()} for d in domain_dict]
    names_str, bounds_str = convert_domain_dict_strings(domain_dict[0])
    confidence_value = float(confidence[0])

    names = names_str.split("|") if names_str else []
    bounds = bounds_str.split("|") if bounds_str else []
    if len(names) != len(bounds):
        raise ValueError(f"Mismatched domain names ({names}) and bounds ({bounds})")

    # gather segments into domains, keyed by domain id
    domains_by_id: dict[str, Domain] = {}
    for domain_id, chopping_by_index in zip(names, bounds, strict=True):
        start_index, end_index = chopping_by_index.split("-")
        segment = Segment(domain_id, int(start_index), int(end_index))
        domains_by_id.setdefault(domain_id, Domain(domain_id)).segments.append(segment)

    # sort domains by the start residue of their first segment
    domains = sorted(domains_by_id.values(), key=lambda dom: dom.segments[0].start_index)

    domain_choppings = []
    for dom in domains:
        segs_str = [
            f"{_residue_label(seg.start_index, labels_by_index)}-{_residue_label(seg.end_index, labels_by_index)}"
            for seg in dom.segments
        ]
        segs_index_str = [f"{seg.start_index}-{seg.end_index}" for seg in dom.segments]
        LOG.info("Segments (index to label): %s -> %s", segs_index_str, segs_str)
        # discontinuous segments within a domain are joined with '_'
        domain_choppings.append("_".join(segs_str))

    # domains are joined with ','
    chopping_str = ",".join(domain_choppings) if domain_choppings else None
    runtime = round(perf_counter() - start, 3)
    result = PredictionResult(
        pdb_path=structure_path,
        sequence_md5=model_structure_md5,
        nres=len(model_structure_seq),
        ndom=len(domain_choppings),
        chopping=chopping_str,
        confidence=confidence_value,
        time_sec=runtime,
    )

    LOG.info("Runtime: %ss", runtime)
    return result


def _render_pymol_image(result: PredictionResult, outer_save_dir: Path) -> None:
    generate_pymol_image(
        pdb_path=str(result.pdb_path),
        chopping=result.chopping or "",
        image_out_path=str(outer_save_dir / f"{result.pdb_path.stem}.png"),
        path_to_script=str(outer_save_dir / "image_gen.pml"),
        pymol_executable=constants.PYMOL_EXE,
    )


def _predict_and_record(
    model: Any,
    structure_path: Path,
    results_file: PredictionResultsFile,
    *,
    pdb_chain_id: str | None,
    renumber_pdbs: bool,
    pymol_visual: bool,
    outer_save_dir: Path,
) -> None:
    """Predict one structure, persist its result, and optionally render it."""
    result = predict(
        model,
        structure_path,
        pdbchain=pdb_chain_id,
        renumber_pdbs=renumber_pdbs,
    )
    results_file.add_result(result)
    if pymol_visual:
        _render_pymol_image(result, outer_save_dir)


def run(args: argparse.Namespace) -> None:
    """Run predictions for the selected input method and write them to the output file."""
    outer_save_dir = Path(args.save_dir).expanduser()
    requested_chain = getattr(args, "chain", None)
    pdb_chain_id = requested_chain if requested_chain is not None else (None if args.use_first_chain else "A")

    input_method = get_input_method(args)
    if input_method not in {"structure_directory", "structure_file"}:
        raise ValueError(f"Input method '{input_method}' is not supported")

    model = load_model(
        model_dir=args.model_dir,
        remove_disordered_domain_threshold=args.remove_disordered_domain_threshold,
        min_ss_components=args.min_ss_components,
        min_domain_length=args.min_domain_length,
        post_process_domains=args.post_process_domains,
        compile_model=getattr(args, "compile_model", False),
        force_cpu=getattr(args, "force_cpu", False),
        use_mps_if_available=getattr(args, "use_mps_if_available", False),
    )
    outer_save_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).expanduser().absolute()

    results_file = PredictionResultsFile(
        csv_path=output_path,
        # args.allow_append means both allow-append and allow-skip
        allow_append=args.allow_append,
        allow_skip=args.allow_append,
    )

    try:
        if input_method == "structure_directory":
            structure_directory = Path(args.structure_directory).expanduser()
            if not structure_directory.is_dir():
                raise NotADirectoryError(structure_directory)
            for entry in sorted(structure_directory.iterdir(), key=lambda path: path.name.casefold()):
                if not entry.is_file() or entry.suffix.lower() not in ACCEPTED_STRUCTURE_FILE_SUFFIXES:
                    LOG.debug("Skipping file %s (suffix %s)", entry.name, entry.suffix)
                    continue

                chain_id = entry.stem
                if results_file.has_result_for_chain_id(chain_id):
                    LOG.info("Skipping file %s (result for '%s' already exists)", entry.name, chain_id)
                    continue

                LOG.info("Making prediction for file %s (chain '%s')", entry.name, chain_id)
                _predict_and_record(
                    model,
                    entry,
                    results_file,
                    pdb_chain_id=pdb_chain_id,
                    renumber_pdbs=args.renumber_pdbs,
                    pymol_visual=args.pymol_visual,
                    outer_save_dir=outer_save_dir,
                )
        else:
            structure_file = Path(args.structure_file).expanduser()
            if not structure_file.is_file():
                raise FileNotFoundError(structure_file)
            if structure_file.suffix.lower() not in ACCEPTED_STRUCTURE_FILE_SUFFIXES:
                raise ValueError(f"Unsupported structure file type: '{structure_file.suffix}'")
            _predict_and_record(
                model,
                structure_file,
                results_file,
                pdb_chain_id=pdb_chain_id,
                renumber_pdbs=args.renumber_pdbs,
                pymol_visual=args.pymol_visual,
                outer_save_dir=outer_save_dir,
            )
    finally:
        # Preserve completed predictions if a later structure fails.
        results_file.flush()
    LOG.info("DONE")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict protein-domain boundaries from PDB or mmCIF structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        type=Path,
        default=constants.DEFAULT_MODEL_DIR,
        help="path to model directory; must contain weights.pt and config.json",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="write results to this TSV file")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--structure-directory",
        "--structure_directory",
        type=Path,
        default=None,
        help="path to a directory of PDB or mmCIF files",
    )
    inputs.add_argument(
        "--structure-file",
        "--structure_file",
        type=Path,
        default=None,
        help="path to a PDB or mmCIF file",
    )
    parser.add_argument(
        "--append",
        "-a",
        dest="allow_append",
        action="store_true",
        default=False,
        help="allow results to be appended to an existing file",
    )
    parser.add_argument(
        "--save-dir",
        "--save_dir",
        type=Path,
        default=Path("results"),
        help="path where optional images are saved",
    )
    parser.add_argument(
        "--no-post-processing",
        "--no_post_processing",
        dest="post_process_domains",
        action="store_false",
        help="disable domain post-processing",
    )
    parser.add_argument(
        "--remove-disordered-domain-threshold",
        "--remove_disordered_domain_threshold",
        type=float,
        default=0.35,
        help="remove domains with less than this proportion of secondary structure",
    )
    parser.add_argument(
        "--min-domain-length",
        "--min_domain_length",
        type=int,
        default=30,
        help="remove domains with fewer residues than this",
    )
    parser.add_argument(
        "--min-ss-components",
        "--min_ss_components",
        type=int,
        default=2,
        help="remove domains with fewer than this number of distinct secondary-structure components",
    )
    parser.add_argument(
        "--pymol-visual",
        "--pymol_visual",
        dest="pymol_visual",
        action="store_true",
        help="generate PyMOL images",
    )
    parser.add_argument("--chain", help="structure chain id to predict; overrides --use-first-chain")
    parser.add_argument(
        "--use-first-chain",
        "--use_first_chain",
        default=True,
        action=argparse.BooleanOptionalAction,
        help='use the first chain in the structure (rather than "A")',
    )
    parser.add_argument(
        "--renumber-pdbs",
        "--renumber_pdbs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="renumber a temporary PDB copy before running STRIDE",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="compile the model with torch.compile (usually worthwhile only for repeated predictions)",
    )
    parser.add_argument("--force-cpu", action="store_true", help="disable accelerator use")
    parser.add_argument(
        "--use-mps-if-available",
        action="store_true",
        help="use Apple Metal acceleration when available",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point."""
    setup_logging()
    run(parse_args(argv))


if __name__ == "__main__":
    main()
