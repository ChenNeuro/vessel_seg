"""基于模板和几何启发式的冠脉语义拓扑赋值。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import Iterable


@dataclass(frozen=True)
class CoronaryTemplateNode:
    """预设冠脉模板中的单个节点定义。"""

    name: str
    parent_name: str | None
    system_name: str
    required: bool = False
    repeatable: bool = False
    description: str = ""


@dataclass(frozen=True)
class AnatomicalFrame:
    """病例内归一化几何坐标系。"""

    origin_mm: tuple[float, float, float]
    left_right_axis: tuple[float, float, float]
    longitudinal_axis: tuple[float, float, float]
    normal_axis: tuple[float, float, float]
    scale_mm: float
    source: str


@dataclass(frozen=True)
class SemanticGeometryFamilyPrior:
    """单个语义家族的几何先验。"""

    family: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    sample_count: int

    def score(self, feature_vector: tuple[float, ...]) -> float:
        if not feature_vector or len(feature_vector) != len(self.mean):
            return 0.5
        penalties = []
        for value, mean, std in zip(feature_vector, self.mean, self.std):
            sigma = max(float(std), 0.15)
            z = (float(value) - float(mean)) / sigma
            penalties.append(min(z * z, 16.0))
        if not penalties:
            return 0.5
        return float(math.exp(-0.5 * sum(penalties) / len(penalties)))


@dataclass(frozen=True)
class SemanticGeometryPrior:
    """跨病例拟合得到的语义几何先验。"""

    version: str
    feature_names: tuple[str, ...]
    families: tuple[SemanticGeometryFamilyPrior, ...]

    def family_prior(self, family: str) -> SemanticGeometryFamilyPrior | None:
        for item in self.families:
            if item.family == family:
                return item
        return None

    def score(self, family: str, feature_vector: tuple[float, ...]) -> float:
        prior = self.family_prior(family)
        if prior is None:
            return 0.5
        return prior.score(feature_vector)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "families": [asdict(item) for item in self.families],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SemanticGeometryPrior":
        families = []
        for item in list(payload.get("families", [])):
            families.append(
                SemanticGeometryFamilyPrior(
                    family=str(item["family"]),
                    feature_names=tuple(str(name) for name in item.get("feature_names", payload.get("feature_names", []))),
                    mean=tuple(float(value) for value in item.get("mean", [])),
                    std=tuple(float(value) for value in item.get("std", [])),
                    sample_count=int(item.get("sample_count", 0)),
                )
            )
        feature_names = tuple(str(name) for name in payload.get("feature_names", []))
        return cls(
            version=str(payload.get("version", "semantic_geometry_prior_v1")),
            feature_names=feature_names,
            families=tuple(families),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SemanticGeometryPrior":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SemanticBranchAssignment:
    """单个分支的语义拓扑赋值。"""

    branch_id: int
    canonical_name: str | None
    semantic_name: str
    semantic_family: str
    semantic_parent_name: str | None
    system_name: str
    path_role: str
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SemanticConsistencyMetrics:
    """基于预设模板的一致性评分。"""

    score: float
    expected_root_count: int
    observed_root_count: int
    root_score: float
    required_coverage: float
    geometry_score: float
    unmatched_branches: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SemanticTopologyResult:
    """整棵树的语义拓扑结果。"""

    version: str
    frame: AnatomicalFrame
    dominance: str
    template_nodes: tuple[CoronaryTemplateNode, ...]
    assignments: tuple[SemanticBranchAssignment, ...]
    consistency: SemanticConsistencyMetrics

    def branch_map(self) -> dict[int, SemanticBranchAssignment]:
        return {assignment.branch_id: assignment for assignment in self.assignments}

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "frame": asdict(self.frame),
            "dominance": self.dominance,
            "template_nodes": [asdict(node) for node in self.template_nodes],
            "assignments": [asdict(assignment) for assignment in self.assignments],
            "assignment_map": {
                str(assignment.branch_id): assignment.semantic_name for assignment in self.assignments
            },
            "consistency": asdict(self.consistency),
        }


@dataclass(frozen=True)
class _BranchDescriptor:
    """内部几何描述子。"""

    branch_id: int
    parent_id: int | None
    child_ids: tuple[int, ...]
    depth: int
    length_mm: float
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    centroid: tuple[float, float, float]
    direction_vec: tuple[float, float, float]
    lambda_pos: float | None
    theta_deg: float | None
    phi_deg: float | None
    subtree_branch_count: int
    subtree_total_length_mm: float
    canonical_name: str | None


_GEOMETRY_FEATURE_NAMES: tuple[str, ...] = (
    "centroid_lr",
    "centroid_long",
    "centroid_norm",
    "direction_lr",
    "direction_long",
    "direction_norm",
    "length_norm",
    "lambda_pos",
    "depth_norm",
    "subtree_length_norm",
)


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vec))
    if norm <= 1e-8:
        return (0.0, 0.0, 1.0)
    return tuple(component / norm for component in vec)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _attachment(branch: dict[str, object]) -> dict[str, object]:
    return dict(branch.get("attachment") or {})


def _attachment_parent(branch: dict[str, object]) -> int | None:
    parent = _attachment(branch).get("parent")
    return None if parent is None else int(parent)


def _branch_length(branch: dict[str, object]) -> float:
    return float(branch.get("length_mm") or 0.0)


def _as_point(values: object) -> tuple[float, float, float]:
    if not isinstance(values, list | tuple) or len(values) < 3:
        return (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _branch_direction(branch: dict[str, object]) -> tuple[float, float, float]:
    start = _as_point(branch.get("start"))
    end = _as_point(branch.get("end"))
    delta = _sub(end, start)
    if math.sqrt(_dot(delta, delta)) <= 1e-8:
        centroid = _as_point(branch.get("centroid"))
        delta = _sub(centroid, start)
    return _normalize(delta)


def _compute_children(branches_by_id: dict[int, dict[str, object]]) -> dict[int, list[int]]:
    children = {branch_id: [] for branch_id in branches_by_id}
    for branch_id, branch in branches_by_id.items():
        parent = _attachment_parent(branch)
        if parent is not None and parent in children:
            children[parent].append(branch_id)
    return children


def _compute_depths(
    roots: list[int],
    children: dict[int, list[int]],
) -> dict[int, int]:
    depths: dict[int, int] = {}
    stack = [(root_id, 0) for root_id in roots]
    while stack:
        branch_id, depth = stack.pop()
        if branch_id in depths and depths[branch_id] <= depth:
            continue
        depths[branch_id] = depth
        for child_id in children.get(branch_id, []):
            stack.append((child_id, depth + 1))
    return depths


def _compute_subtree_stats(
    branch_id: int,
    branches_by_id: dict[int, dict[str, object]],
    children: dict[int, list[int]],
    cache: dict[int, tuple[int, float]],
) -> tuple[int, float]:
    cached = cache.get(branch_id)
    if cached is not None:
        return cached
    count = 1
    total_length = _branch_length(branches_by_id[branch_id])
    for child_id in children.get(branch_id, []):
        child_count, child_total_length = _compute_subtree_stats(child_id, branches_by_id, children, cache)
        count += child_count
        total_length += child_total_length
    result = (count, total_length)
    cache[branch_id] = result
    return result


def _default_template() -> tuple[CoronaryTemplateNode, ...]:
    return (
        CoronaryTemplateNode("LeftMain", None, "LCA", required=True, description="左主干"),
        CoronaryTemplateNode("LeftAnteriorDescending", "LeftMain", "LCA", required=True, description="前降支主干"),
        CoronaryTemplateNode("LeftCircumflex", "LeftMain", "LCA", required=True, description="回旋支主干"),
        CoronaryTemplateNode("RamusIntermedius", "LeftMain", "LCA", repeatable=True, description="中间支"),
        CoronaryTemplateNode("DiagonalBranch", "LeftAnteriorDescending", "LCA", repeatable=True, description="对角支"),
        CoronaryTemplateNode("SeptalPerforator", "LeftAnteriorDescending", "LCA", repeatable=True, description="间隔支"),
        CoronaryTemplateNode("ObtuseMarginal", "LeftCircumflex", "LCA", repeatable=True, description="钝缘支"),
        CoronaryTemplateNode("LeftPosteriorDescending", "LeftCircumflex", "LCA", repeatable=False, description="左侧后降支"),
        CoronaryTemplateNode("RightCoronaryArtery", None, "RCA", required=True, description="右冠主干"),
        CoronaryTemplateNode("ConusBranch", "RightCoronaryArtery", "RCA", repeatable=False, description="圆锥支"),
        CoronaryTemplateNode("SinoatrialNodeArtery", "RightCoronaryArtery", "RCA", repeatable=False, description="窦房结支"),
        CoronaryTemplateNode("RightMarginal", "RightCoronaryArtery", "RCA", repeatable=True, description="右缘支"),
        CoronaryTemplateNode("RightPosteriorDescending", "RightCoronaryArtery", "RCA", repeatable=False, description="右侧后降支"),
        CoronaryTemplateNode("PosterolateralBranch", "RightCoronaryArtery", "RCA", repeatable=True, description="后外侧支"),
        CoronaryTemplateNode("AccessoryBranch", None, "AUX", repeatable=True, description="无法可靠映射的附属分支"),
    )


def _build_descriptors(tree_payload: dict[str, object]) -> tuple[dict[int, _BranchDescriptor], list[int]]:
    branches = list(tree_payload.get("branches", []))
    branches_by_id = {int(branch["branch_id"]): branch for branch in branches}
    roots = [int(root_id) for root_id in tree_payload.get("roots", [])]
    if not roots:
        roots = [branch_id for branch_id, branch in branches_by_id.items() if _attachment_parent(branch) is None]
    children = _compute_children(branches_by_id)
    subtree_stats: dict[int, tuple[int, float]] = {}
    for branch_id in branches_by_id:
        _compute_subtree_stats(branch_id, branches_by_id, children, subtree_stats)
    depths = _compute_depths(roots, children)

    descriptors: dict[int, _BranchDescriptor] = {}
    for branch_id, branch in branches_by_id.items():
        attachment = _attachment(branch)
        naming = dict(branch.get("naming") or {})
        subtree_count, subtree_total_length = subtree_stats[branch_id]
        descriptors[branch_id] = _BranchDescriptor(
            branch_id=branch_id,
            parent_id=_attachment_parent(branch),
            child_ids=tuple(children.get(branch_id, [])),
            depth=int(depths.get(branch_id, 0)),
            length_mm=_branch_length(branch),
            start=_as_point(branch.get("start")),
            end=_as_point(branch.get("end")),
            centroid=_as_point(branch.get("centroid")),
            direction_vec=_branch_direction(branch),
            lambda_pos=None if attachment.get("lambda_pos") is None else float(attachment.get("lambda_pos")),
            theta_deg=None if attachment.get("theta_deg") is None else float(attachment.get("theta_deg")),
            phi_deg=None if attachment.get("phi_deg") is None else float(attachment.get("phi_deg")),
            subtree_branch_count=int(subtree_count),
            subtree_total_length_mm=float(subtree_total_length),
            canonical_name=str(naming.get("canonical_name")) if naming.get("canonical_name") is not None else None,
        )
    return descriptors, roots


def _infer_root_systems(
    roots: list[int],
    descriptors: dict[int, _BranchDescriptor],
) -> dict[int, str]:
    if not roots:
        return {}
    if len(roots) == 1:
        return {roots[0]: "MAIN"}

    scored = sorted(
        roots,
        key=lambda branch_id: (
            -descriptors[branch_id].subtree_branch_count,
            -descriptors[branch_id].subtree_total_length_mm,
            -descriptors[branch_id].length_mm,
            float(branch_id),
        ),
    )
    systems: dict[int, str] = {}
    for index, root_id in enumerate(scored):
        if index == 0:
            systems[root_id] = "LCA"
        elif index == 1:
            systems[root_id] = "RCA"
        else:
            systems[root_id] = f"AUX_{index - 1:02d}"
    return systems


def _build_frame(
    root_systems: dict[int, str],
    descriptors: dict[int, _BranchDescriptor],
) -> AnatomicalFrame:
    lca_root = next((root_id for root_id, system_name in root_systems.items() if system_name == "LCA"), None)
    rca_root = next((root_id for root_id, system_name in root_systems.items() if system_name == "RCA"), None)
    if lca_root is not None and rca_root is not None:
        lca_center = descriptors[lca_root].centroid
        rca_center = descriptors[rca_root].centroid
        origin = _scale(_add(lca_center, rca_center), 0.5)
        scale_mm = max(10.0, math.sqrt(_dot(_sub(lca_center, rca_center), _sub(lca_center, rca_center))))
        left_right_axis = _normalize(_sub(lca_center, rca_center))
        forward_seed = _normalize(_add(descriptors[lca_root].direction_vec, descriptors[rca_root].direction_vec))
        normal_axis = _normalize(_cross(left_right_axis, forward_seed))
        if math.sqrt(_dot(normal_axis, normal_axis)) <= 1e-8:
            normal_axis = (0.0, 0.0, 1.0)
        longitudinal_axis = _normalize(_cross(normal_axis, left_right_axis))
        return AnatomicalFrame(
            origin_mm=origin,
            left_right_axis=left_right_axis,
            longitudinal_axis=longitudinal_axis,
            normal_axis=normal_axis,
            scale_mm=scale_mm,
            source="root_pair",
        )

    if descriptors:
        first = next(iter(descriptors.values()))
        origin = first.centroid
        return AnatomicalFrame(
            origin_mm=origin,
            left_right_axis=(1.0, 0.0, 0.0),
            longitudinal_axis=first.direction_vec,
            normal_axis=_normalize(_cross((1.0, 0.0, 0.0), first.direction_vec)),
            scale_mm=max(10.0, first.length_mm),
            source="single_root_fallback",
        )
    return AnatomicalFrame(
        origin_mm=(0.0, 0.0, 0.0),
        left_right_axis=(1.0, 0.0, 0.0),
        longitudinal_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        scale_mm=100.0,
        source="empty_tree",
    )


def _descriptor_feature_vector(
    descriptor: _BranchDescriptor,
    frame: AnatomicalFrame,
) -> tuple[float, ...]:
    scale_mm = max(float(frame.scale_mm), 1.0)
    rel_centroid = _sub(descriptor.centroid, frame.origin_mm)
    return (
        _dot(rel_centroid, frame.left_right_axis) / scale_mm,
        _dot(rel_centroid, frame.longitudinal_axis) / scale_mm,
        _dot(rel_centroid, frame.normal_axis) / scale_mm,
        _dot(descriptor.direction_vec, frame.left_right_axis),
        _dot(descriptor.direction_vec, frame.longitudinal_axis),
        _dot(descriptor.direction_vec, frame.normal_axis),
        descriptor.length_mm / scale_mm,
        0.0 if descriptor.lambda_pos is None else float(descriptor.lambda_pos),
        descriptor.depth / 5.0,
        descriptor.subtree_total_length_mm / scale_mm,
    )


def _fit_family_prior(
    family: str,
    feature_vectors: list[tuple[float, ...]],
) -> SemanticGeometryFamilyPrior:
    means = []
    stds = []
    feature_count = len(_GEOMETRY_FEATURE_NAMES)
    for feature_index in range(feature_count):
        values = [vector[feature_index] for vector in feature_vectors]
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(len(values))
        means.append(float(mean))
        stds.append(float(math.sqrt(variance) + 1e-3))
    return SemanticGeometryFamilyPrior(
        family=family,
        feature_names=_GEOMETRY_FEATURE_NAMES,
        mean=tuple(means),
        std=tuple(stds),
        sample_count=len(feature_vectors),
    )


def fit_semantic_geometry_prior_from_tree_payloads(
    tree_payloads: Iterable[dict[str, object]],
    *,
    min_samples_per_family: int = 3,
) -> SemanticGeometryPrior:
    """从已有语义树拟合几何先验。"""
    family_vectors: dict[str, list[tuple[float, ...]]] = {}
    for tree_payload in tree_payloads:
        descriptors, roots = _build_descriptors(tree_payload)
        if not descriptors:
            continue
        frame = _build_frame(_infer_root_systems(roots, descriptors), descriptors)
        for branch in list(tree_payload.get("branches", [])):
            semantic = dict(branch.get("semantic") or {})
            family = semantic.get("semantic_family")
            if not family or family == "AccessoryBranch":
                continue
            branch_id = int(branch["branch_id"])
            family_vectors.setdefault(str(family), []).append(_descriptor_feature_vector(descriptors[branch_id], frame))

    priors = []
    for family, vectors in sorted(family_vectors.items()):
        if len(vectors) < max(1, int(min_samples_per_family)):
            continue
        priors.append(_fit_family_prior(family, vectors))
    return SemanticGeometryPrior(
        version="semantic_geometry_prior_v1",
        feature_names=_GEOMETRY_FEATURE_NAMES,
        families=tuple(priors),
    )


def fit_semantic_geometry_prior_from_tree_paths(
    tree_paths: Iterable[Path],
    *,
    min_samples_per_family: int = 3,
) -> SemanticGeometryPrior:
    """从一组 tree.json 路径拟合几何先验。"""
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in tree_paths if path.exists()]
    return fit_semantic_geometry_prior_from_tree_payloads(payloads, min_samples_per_family=min_samples_per_family)


def _scaled_metric(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.5
    return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))


def _continuation_scores(
    parent_id: int,
    child_ids: list[int],
    descriptors: dict[int, _BranchDescriptor],
) -> dict[int, float]:
    if not child_ids:
        return {}
    parent = descriptors[parent_id]
    max_subtree = max(descriptors[child_id].subtree_branch_count for child_id in child_ids)
    min_subtree = min(descriptors[child_id].subtree_branch_count for child_id in child_ids)
    max_length = max(descriptors[child_id].subtree_total_length_mm for child_id in child_ids)
    min_length = min(descriptors[child_id].subtree_total_length_mm for child_id in child_ids)
    scores: dict[int, float] = {}
    for child_id in child_ids:
        child = descriptors[child_id]
        direction_score = 0.5 * (_dot(parent.direction_vec, child.direction_vec) + 1.0)
        subtree_score = _scaled_metric(child.subtree_branch_count, min_subtree, max_subtree)
        length_score = _scaled_metric(child.subtree_total_length_mm, min_length, max_length)
        scores[child_id] = 0.55 * direction_score + 0.25 * subtree_score + 0.20 * length_score
    return scores


def _semantic_confidence(base_score: float, bonus: float = 0.0) -> float:
    return max(0.35, min(0.98, 0.45 + 0.45 * float(base_score) + bonus))


def assign_semantic_topology(
    tree_payload: dict[str, object],
    geometry_prior: SemanticGeometryPrior | None = None,
) -> SemanticTopologyResult:
    """为 clean tree 赋予基于模板的语义拓扑标签。"""
    descriptors, roots = _build_descriptors(tree_payload)
    root_systems = _infer_root_systems(roots, descriptors)
    frame = _build_frame(root_systems, descriptors)
    template_nodes = _default_template()

    assignment_by_branch: dict[int, SemanticBranchAssignment] = {}
    family_counters: dict[str, int] = {}
    warnings: list[str] = []

    feature_vectors = {
        branch_id: _descriptor_feature_vector(descriptor, frame)
        for branch_id, descriptor in descriptors.items()
    }

    def family_prior_score(family: str, branch_id: int) -> float:
        if geometry_prior is None:
            return 0.5
        return geometry_prior.score(family, feature_vectors[branch_id])

    def next_semantic_name(family: str) -> str:
        family_counters[family] = family_counters.get(family, 0) + 1
        return f"{family}.{family_counters[family]:02d}"

    def assign_branch(
        branch_id: int,
        *,
        family: str,
        system_name: str,
        path_role: str,
        semantic_parent_name: str | None,
        confidence: float,
        reasons: list[str],
    ) -> str:
        descriptor = descriptors[branch_id]
        semantic_name = next_semantic_name(family)
        assignment_by_branch[branch_id] = SemanticBranchAssignment(
            branch_id=branch_id,
            canonical_name=descriptor.canonical_name,
            semantic_name=semantic_name,
            semantic_family=family,
            semantic_parent_name=semantic_parent_name,
            system_name=system_name,
            path_role=path_role,
            confidence=confidence,
            reasons=tuple(reasons),
        )
        return semantic_name

    def unassigned_children(branch_id: int) -> list[int]:
        return [child_id for child_id in descriptors[branch_id].child_ids if child_id not in assignment_by_branch]

    def classify_lad_side(parent_id: int, child_id: int, system_name: str) -> tuple[str, float, list[str]]:
        parent_descriptor = descriptors[parent_id]
        descriptor = descriptors[child_id]
        lateral_projection = _dot(_sub(descriptor.centroid, parent_descriptor.centroid), frame.left_right_axis)
        diagonal_prior = family_prior_score("DiagonalBranch", child_id)
        septal_prior = family_prior_score("SeptalPerforator", child_id)
        prior_prefers_diagonal = diagonal_prior > septal_prior + 0.08
        geometry_prefers_diagonal = lateral_projection >= 0.0
        if system_name == "LCA" and (geometry_prefers_diagonal or prior_prefers_diagonal):
            return (
                "DiagonalBranch",
                _semantic_confidence(0.55 + 0.35 * diagonal_prior, bonus=0.08),
                [
                    f"lateral_projection={lateral_projection:.3f}",
                    f"diagonal_prior={diagonal_prior:.3f}",
                    "lad_side=diagonal",
                ],
            )
        return (
            "SeptalPerforator",
            _semantic_confidence(0.52 + 0.35 * septal_prior, bonus=0.04),
            [
                f"lateral_projection={lateral_projection:.3f}",
                f"septal_prior={septal_prior:.3f}",
                "lad_side=septal",
            ],
        )

    rca_state = {
        "conus_assigned": False,
        "san_assigned": False,
        "pda_assigned": False,
        "plb_assigned": False,
    }

    def classify_rca_side(_parent_id: int, child_id: int) -> tuple[str, float, list[str]]:
        descriptor = descriptors[child_id]
        lambda_pos = descriptor.lambda_pos if descriptor.lambda_pos is not None else 0.5
        conus_prior = family_prior_score("ConusBranch", child_id)
        san_prior = family_prior_score("SinoatrialNodeArtery", child_id)
        rpda_prior = family_prior_score("RightPosteriorDescending", child_id)
        plb_prior = family_prior_score("PosterolateralBranch", child_id)
        marginal_prior = family_prior_score("RightMarginal", child_id)
        if (lambda_pos <= 0.20 or conus_prior >= 0.62) and not rca_state["conus_assigned"]:
            rca_state["conus_assigned"] = True
            return (
                "ConusBranch",
                _semantic_confidence(0.45 + 0.40 * conus_prior),
                [f"lambda={lambda_pos:.3f}", f"conus_prior={conus_prior:.3f}", "rca_side=conus"],
            )
        if (lambda_pos <= 0.35 or san_prior >= 0.62) and not rca_state["san_assigned"]:
            rca_state["san_assigned"] = True
            return (
                "SinoatrialNodeArtery",
                _semantic_confidence(0.42 + 0.40 * san_prior),
                [f"lambda={lambda_pos:.3f}", f"san_prior={san_prior:.3f}", "rca_side=sa_node"],
            )
        distal_score = 0.5 * float(descriptor.depth >= 2) + 0.5 * lambda_pos
        if (distal_score >= 0.75 or rpda_prior >= 0.62) and not rca_state["pda_assigned"]:
            rca_state["pda_assigned"] = True
            return (
                "RightPosteriorDescending",
                _semantic_confidence(0.44 + 0.40 * rpda_prior),
                [f"lambda={lambda_pos:.3f}", f"depth={descriptor.depth}", f"rpda_prior={rpda_prior:.3f}", "rca_side=rpda"],
            )
        if (distal_score >= 0.60 or plb_prior >= 0.62) and not rca_state["plb_assigned"]:
            rca_state["plb_assigned"] = True
            return (
                "PosterolateralBranch",
                _semantic_confidence(0.40 + 0.40 * plb_prior),
                [f"lambda={lambda_pos:.3f}", f"depth={descriptor.depth}", f"plb_prior={plb_prior:.3f}", "rca_side=plb"],
            )
        return (
            "RightMarginal",
            _semantic_confidence(0.38 + 0.40 * marginal_prior),
            [f"lambda={lambda_pos:.3f}", f"depth={descriptor.depth}", f"marginal_prior={marginal_prior:.3f}", "rca_side=marginal"],
        )

    def follow_path(
        seed_id: int,
        *,
        family: str,
        system_name: str,
        semantic_parent_name: str | None,
        seed_reason: list[str],
        side_classifier,
    ) -> None:
        current_id = seed_id
        current_parent_semantic = semantic_parent_name
        first_segment = True
        while current_id not in assignment_by_branch:
            semantic_name = assign_branch(
                current_id,
                family=family,
                system_name=system_name,
                path_role="trunk_seed" if first_segment else "continuation",
                semantic_parent_name=current_parent_semantic,
                confidence=_semantic_confidence(0.82 if first_segment else 0.74),
                reasons=list(seed_reason if first_segment else [f"continuation_of={family}"]),
            )
            first_segment = False
            children_ids = unassigned_children(current_id)
            if not children_ids:
                break
            continuation_scores = _continuation_scores(current_id, children_ids, descriptors)
            continuation_id = max(
                children_ids,
                key=lambda child_id: 0.55 * continuation_scores.get(child_id, 0.5) + 0.45 * family_prior_score(family, child_id),
            )
            side_ids = [child_id for child_id in children_ids if child_id != continuation_id]
            for side_id in side_ids:
                side_family, side_confidence, side_reasons = side_classifier(current_id, side_id)
                follow_path(
                    side_id,
                    family=side_family,
                    system_name=system_name,
                    semantic_parent_name=semantic_name,
                    seed_reason=side_reasons + [f"seed_parent={semantic_name}"],
                    side_classifier=side_classifier if side_family == family else _classifier_for_family(side_family, system_name),
                )
            current_parent_semantic = semantic_name
            current_id = continuation_id

    def _classifier_for_family(family: str, system_name: str):
        if family == "LeftAnteriorDescending":
            return lambda parent_id, child_id: classify_lad_side(parent_id, child_id, system_name)
        if family == "LeftCircumflex":
            return lambda _parent_id, _child_id: (
                "ObtuseMarginal",
                _semantic_confidence(0.58),
                [f"seed_family={family}", "lcx_side=obtuse_marginal"],
            )
        if family == "RamusIntermedius":
            return lambda _parent_id, _child_id: (
                "DiagonalBranch",
                _semantic_confidence(0.50),
                [f"seed_family={family}", "ri_side=diagonal_like"],
            )
        if family == "RightCoronaryArtery":
            return classify_rca_side
        return lambda _parent_id, _child_id: (
            "AccessoryBranch",
            _semantic_confidence(0.42),
            [f"seed_family={family}", "fallback=accessory"],
        )

    ordered_roots = sorted(
        roots,
        key=lambda branch_id: (
            0 if root_systems.get(branch_id) == "LCA" else 1,
            0 if root_systems.get(branch_id) == "RCA" else 1,
            float(branch_id),
        ),
    )

    dominance = "Unknown"
    for root_id in ordered_roots:
        system_name = root_systems.get(root_id, "AUX")
        descriptor = descriptors[root_id]
        if system_name == "LCA":
            root_name = assign_branch(
                root_id,
                family="LeftMain",
                system_name=system_name,
                path_role="root",
                semantic_parent_name=None,
                confidence=_semantic_confidence(0.88, bonus=0.06),
                reasons=["root_system=LCA", f"subtree={descriptor.subtree_branch_count}"],
            )
            child_ids = unassigned_children(root_id)
            if not child_ids:
                warnings.append("LCA root has no children; LAD/LCx cannot be resolved.")
                continue
            child_scores = _continuation_scores(root_id, child_ids, descriptors)
            sibling_max_subtree = max(descriptors[child_id].subtree_branch_count for child_id in child_ids)
            sibling_min_subtree = min(descriptors[child_id].subtree_branch_count for child_id in child_ids)

            def lca_seed_score(family: str, child_id: int) -> float:
                prior_score = family_prior_score(family, child_id)
                subtree_score = _scaled_metric(descriptors[child_id].subtree_branch_count, sibling_min_subtree, sibling_max_subtree)
                continuity_score = child_scores.get(child_id, 0.5)
                if family == "LeftAnteriorDescending":
                    return 0.45 * prior_score + 0.35 * continuity_score + 0.20 * subtree_score
                if family == "LeftCircumflex":
                    return 0.55 * prior_score + 0.20 * (1.0 - continuity_score) + 0.25 * subtree_score
                if family == "RamusIntermedius":
                    return 0.60 * prior_score + 0.20 * (1.0 - subtree_score) + 0.20 * (1.0 - continuity_score)
                return prior_score

            lad_seed = max(child_ids, key=lambda child_id: lca_seed_score("LeftAnteriorDescending", child_id))
            follow_path(
                lad_seed,
                family="LeftAnteriorDescending",
                system_name=system_name,
                semantic_parent_name=root_name,
                seed_reason=[
                    "lca_major_child=lad",
                    f"family_score={lca_seed_score('LeftAnteriorDescending', lad_seed):.3f}",
                    f"prior={family_prior_score('LeftAnteriorDescending', lad_seed):.3f}",
                ],
                side_classifier=_classifier_for_family("LeftAnteriorDescending", system_name),
            )
            remaining_children = [child_id for child_id in child_ids if child_id != lad_seed]
            if remaining_children:
                lcx_seed = max(remaining_children, key=lambda child_id: lca_seed_score("LeftCircumflex", child_id))
                follow_path(
                    lcx_seed,
                    family="LeftCircumflex",
                    system_name=system_name,
                    semantic_parent_name=root_name,
                    seed_reason=[
                        "lca_second_child=lcx",
                        f"family_score={lca_seed_score('LeftCircumflex', lcx_seed):.3f}",
                        f"prior={family_prior_score('LeftCircumflex', lcx_seed):.3f}",
                    ],
                    side_classifier=_classifier_for_family("LeftCircumflex", system_name),
                )
            for extra_id in sorted(
                [child_id for child_id in child_ids if child_id not in {lad_seed, *([] if not remaining_children else [lcx_seed])}],
                key=lambda child_id: (-lca_seed_score("RamusIntermedius", child_id), float(child_id)),
            ):
                follow_path(
                    extra_id,
                    family="RamusIntermedius",
                    system_name=system_name,
                    semantic_parent_name=root_name,
                    seed_reason=[
                        "lca_extra_child=ri",
                        f"family_score={lca_seed_score('RamusIntermedius', extra_id):.3f}",
                        f"prior={family_prior_score('RamusIntermedius', extra_id):.3f}",
                    ],
                    side_classifier=_classifier_for_family("RamusIntermedius", system_name),
                )
        elif system_name == "RCA":
            dominance = "Right"
            follow_path(
                root_id,
                family="RightCoronaryArtery",
                system_name=system_name,
                semantic_parent_name=None,
                seed_reason=[
                    "root_system=RCA",
                    f"subtree={descriptor.subtree_branch_count}",
                    f"prior={family_prior_score('RightCoronaryArtery', root_id):.3f}",
                ],
                side_classifier=_classifier_for_family("RightCoronaryArtery", system_name),
            )
        else:
            follow_path(
                root_id,
                family="AccessoryBranch",
                system_name=system_name,
                semantic_parent_name=None,
                seed_reason=[f"root_system={system_name}", "fallback=accessory"],
                side_classifier=_classifier_for_family("AccessoryBranch", system_name),
            )

    for branch_id in sorted(descriptors):
        if branch_id in assignment_by_branch:
            continue
        parent_id = descriptors[branch_id].parent_id
        parent_assignment = assignment_by_branch.get(parent_id) if parent_id is not None else None
        assign_branch(
            branch_id,
            family="AccessoryBranch",
            system_name=parent_assignment.system_name if parent_assignment is not None else "AUX",
            path_role="unmatched",
            semantic_parent_name=None if parent_assignment is None else parent_assignment.semantic_name,
            confidence=_semantic_confidence(0.30),
            reasons=["unmatched_after_template_assignment"],
        )

    required_families = {
        node.name for node in template_nodes if node.required and node.system_name in {"LCA", "RCA"}
    }
    assigned_families = {assignment.semantic_family for assignment in assignment_by_branch.values()}
    required_hits = len(required_families & assigned_families)
    required_coverage = required_hits / float(len(required_families)) if required_families else 1.0
    root_score = 1.0 if len(roots) == 2 else (0.7 if len(roots) == 1 else max(0.2, 1.0 - 0.2 * (len(roots) - 2)))
    geometry_score = (
        sum(assignment.confidence for assignment in assignment_by_branch.values()) / float(len(assignment_by_branch))
        if assignment_by_branch
        else 0.0
    )
    unmatched_count = sum(1 for assignment in assignment_by_branch.values() if assignment.semantic_family == "AccessoryBranch")
    score = max(
        0.0,
        min(
            100.0,
            100.0
            * (
                0.35 * root_score
                + 0.35 * required_coverage
                + 0.20 * geometry_score
                + 0.10 * max(0.0, 1.0 - unmatched_count / max(1, len(descriptors)))
            ),
        ),
    )
    if len(roots) != 2:
        warnings.append(f"Observed {len(roots)} roots; expected 2 for typical coronary topology.")
    if "LeftAnteriorDescending" not in assigned_families or "LeftCircumflex" not in assigned_families:
        warnings.append("Left system trunk split is incomplete.")
    if "RightCoronaryArtery" not in assigned_families:
        warnings.append("Right coronary trunk is missing.")

    ordered_assignments = tuple(assignment_by_branch[branch_id] for branch_id in sorted(assignment_by_branch))
    return SemanticTopologyResult(
        version="semantic_topology_v1",
        frame=frame,
        dominance=dominance,
        template_nodes=template_nodes,
        assignments=ordered_assignments,
        consistency=SemanticConsistencyMetrics(
            score=round(score, 4),
            expected_root_count=2,
            observed_root_count=len(roots),
            root_score=round(root_score, 4),
            required_coverage=round(required_coverage, 4),
            geometry_score=round(geometry_score, 4),
            unmatched_branches=int(unmatched_count),
            warnings=tuple(warnings),
        ),
    )


def enrich_tree_with_semantic_topology(
    tree_payload: dict[str, object],
    geometry_prior: SemanticGeometryPrior | None = None,
) -> dict[str, object]:
    """将语义拓扑赋值写回 tree payload。"""
    result = assign_semantic_topology(tree_payload, geometry_prior=geometry_prior)
    branch_map = result.branch_map()
    enriched = dict(tree_payload)
    enriched_branches = []
    for branch in list(tree_payload.get("branches", [])):
        branch_id = int(branch["branch_id"])
        enriched_branch = dict(branch)
        enriched_branch["semantic"] = asdict(branch_map[branch_id])
        enriched_branches.append(enriched_branch)
    enriched["branches"] = enriched_branches
    enriched["semantic_topology"] = result.to_dict()
    return enriched


def apply_semantic_topology(
    tree_json_path: Path,
    semantic_topology_json_path: Path | None = None,
    geometry_prior_path: Path | None = None,
) -> SemanticTopologyResult:
    """读取 tree.json，写回语义拓扑，并导出单独结果。"""
    payload = json.loads(tree_json_path.read_text(encoding="utf-8"))
    geometry_prior = None if geometry_prior_path is None else SemanticGeometryPrior.load(geometry_prior_path)
    result = assign_semantic_topology(payload, geometry_prior=geometry_prior)
    enriched = enrich_tree_with_semantic_topology(payload, geometry_prior=geometry_prior)
    tree_json_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    if semantic_topology_json_path is not None:
        semantic_topology_json_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_topology_json_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return result
