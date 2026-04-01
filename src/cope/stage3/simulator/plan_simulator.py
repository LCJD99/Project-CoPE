import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.cope.stage3.simulator.plan_parser import parse_plan


@dataclass
class RewardResult:
    total: float
    valid: bool
    details: Dict[str, float]


class PlanSimulator:
    def __init__(
        self,
        registry: Dict,
        profiling_path: Optional[str] = None,
        profiling_fallback_path: Optional[str] = None,
    ):
        self.registry = registry
        self.profiling_latency = self._load_profiling_latency(
            profiling_path,
            profiling_fallback_path,
        )

    def _load_profiling_latency(
        self,
        profiling_path: Optional[str],
        profiling_fallback_path: Optional[str],
    ) -> Dict[Tuple[str, str, int, float, int, float], float]:
        paths = [profiling_path, profiling_fallback_path]
        latency_map: Dict[Tuple[str, str, int, float, int, float], float] = {}
        for path in paths:
            if path is None:
                continue
            file_path = Path(path)
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (
                        row["tool"],
                        row["input_size"],
                        int(float(row["cpu_core"])),
                        float(row["cpu_mem_gb"]),
                        int(float(row["gpu_sm"])),
                        float(row["gpu_mem_gb"]),
                    )
                    latency_map[key] = float(row["latency_ms"])
            if latency_map:
                return latency_map
        return latency_map

    def evaluate(
        self,
        plan_text: str,
        system_state: Dict[str, float],
        ground_truth_dag: Optional[Dict] = None,
        baseline_latency_ms: Optional[float] = None,
    ) -> RewardResult:
        try:
            statements = parse_plan(plan_text)
        except ValueError:
            return RewardResult(total=-1.0, valid=False, details={"hard": -1.0})

        dag_result = self._build_generated_dag(statements)
        if dag_result["refs_invalid"]:
            return RewardResult(total=-1.2, valid=False, details={"hard": -1.2})

        if self._has_resource_violation(dag_result["nodes"], system_state):
            return RewardResult(total=-1.6, valid=False, details={"hard": -1.6})

        if self._has_deadlock(dag_result["nodes"], dag_result["edges"]):
            return RewardResult(total=-2.0, valid=False, details={"hard": -2.0})

        q_score = self._compute_correctness_score(
            dag_result["nodes"],
            dag_result["edges"],
            ground_truth_dag,
        )
        if q_score < 1.0:
            task_reward = 0.2 + 0.6 * q_score
            return RewardResult(
                total=task_reward,
                valid=True,
                details={"q": q_score, "task": task_reward},
            )

        latency_reward = self._compute_latency_reward(
            dag_result["nodes"],
            dag_result["edges"],
            baseline_latency_ms,
        )
        total = 1.2 + 0.3 * latency_reward
        return RewardResult(
            total=total,
            valid=True,
            details={"q": 1.0, "latency": latency_reward},
        )

    def _build_generated_dag(self, statements: List[Dict]) -> Dict:
        nodes: Dict[int, Dict] = {}
        edges: Set[Tuple[int, int]] = set()
        refs_invalid = False
        finish_refs: Set[int] = set()

        for stmt in statements:
            if stmt["type"] == "exec":
                ref_idx = self._ref_to_index(stmt["ref"])
                tool_token = stmt["tool"].strip("<>")
                token_info = self.registry.get("tokens", {}).get(tool_token)
                if token_info is None:
                    refs_invalid = True
                    continue

                arg_refs, normalized_args = self._normalize_arguments(stmt["args"])
                wait_refs = [self._ref_to_index(ref) for ref in stmt["wait"]]
                dependency_refs = set(wait_refs + arg_refs)

                nodes[ref_idx] = {
                    "idx": ref_idx,
                    "tool_token": tool_token,
                    "tool_name": token_info.get("tool_name", tool_token.lower()),
                    "input_size": token_info.get("input_size", "small"),
                    "resources": token_info.get("resources", {}),
                    "args": normalized_args,
                    "deps": dependency_refs,
                    "registry_latency": float(token_info.get("latency_ms", 0.0)),
                }

                for dep in dependency_refs:
                    edges.add((dep, ref_idx))

            elif stmt["type"] == "sync":
                for ref in stmt["refs"]:
                    dep = self._ref_to_index(ref)
                    if dep not in nodes:
                        refs_invalid = True

            elif stmt["type"] == "finish":
                refs = stmt.get("refs", [])
                if not refs:
                    refs_invalid = True
                else:
                    finish_refs = {self._ref_to_index(ref) for ref in refs}

        for src, dst in edges:
            if src not in nodes or dst not in nodes:
                refs_invalid = True

        if not finish_refs:
            refs_invalid = True
        else:
            for finish_ref in finish_refs:
                if finish_ref not in nodes:
                    refs_invalid = True

        return {
            "nodes": nodes,
            "edges": edges,
            "refs_invalid": refs_invalid,
        }

    def _normalize_arguments(self, arg_text: str) -> Tuple[List[int], List[str]]:
        inner = arg_text.strip()[1:-1]
        if not inner:
            return [], []
        parts = [part.strip() for part in inner.split(",")]
        ref_indices: List[int] = []
        normalized: List[str] = []
        for part in parts:
            if part.startswith("<REF_") and part.endswith(">"):
                ref_idx = self._ref_to_index(part)
                ref_indices.append(ref_idx)
                normalized.append("<node>")
            elif part.startswith('"') and part.endswith('"'):
                normalized.append(part[1:-1])
            else:
                normalized.append(part)
        return ref_indices, normalized

    def _ref_to_index(self, ref: str) -> int:
        return int(ref.replace("<REF_", "").replace(">", ""))

    def _has_resource_violation(
        self, nodes: Dict[int, Dict], system_state: Dict[str, float]
    ) -> bool:
        cpu_core = system_state.get("cpu_core")
        cpu_memory = system_state.get("cpu_memory")
        gpu_sm = system_state.get("gpu_sm")
        gpu_memory = system_state.get("gpu_memory")
        if None in (cpu_core, cpu_memory, gpu_sm, gpu_memory):
            return True

        for node in nodes.values():
            resources = node["resources"]
            if None in (
                resources.get("cpu_core"),
                resources.get("cpu_mem_gb"),
                resources.get("gpu_sm"),
                resources.get("gpu_mem_gb"),
            ):
                return True
            if (
                resources["cpu_core"] > cpu_core
                or resources["cpu_mem_gb"] > cpu_memory
                or resources["gpu_sm"] > gpu_sm
                or resources["gpu_mem_gb"] > gpu_memory
            ):
                return True
        return False

    def _has_deadlock(
        self, nodes: Dict[int, Dict], edges: Set[Tuple[int, int]]
    ) -> bool:
        indegree = {idx: 0 for idx in nodes}
        outgoing: Dict[int, List[int]] = {idx: [] for idx in nodes}
        for src, dst in edges:
            indegree[dst] += 1
            outgoing[src].append(dst)

        queue = [idx for idx, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for nxt in outgoing[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return visited != len(nodes)

    def _compute_correctness_score(
        self,
        generated_nodes: Dict[int, Dict],
        generated_edges: Set[Tuple[int, int]],
        ground_truth_dag: Optional[Dict],
    ) -> float:
        if not ground_truth_dag:
            return 0.0
        gt_nodes = ground_truth_dag.get("nodes", [])
        gt_links = ground_truth_dag.get("links", [])

        gen_node_signatures = {
            self._node_signature(node["tool_name"], node["args"])
            for node in generated_nodes.values()
        }
        gt_node_signatures = {
            self._node_signature(node.get("tool", ""), node.get("arguments", []))
            for node in gt_nodes
        }

        index_to_signature = {
            int(node.get("index", i)): self._node_signature(
                node.get("tool", ""),
                node.get("arguments", []),
            )
            for i, node in enumerate(gt_nodes)
        }
        gen_idx_to_signature = {
            idx: self._node_signature(node["tool_name"], node["args"])
            for idx, node in generated_nodes.items()
        }

        gen_edge_signatures = {
            (gen_idx_to_signature[src], gen_idx_to_signature[dst])
            for src, dst in generated_edges
            if src in gen_idx_to_signature and dst in gen_idx_to_signature
        }
        gt_edge_signatures = {
            (
                index_to_signature.get(link.get("from")),
                index_to_signature.get(link.get("to")),
            )
            for link in gt_links
            if index_to_signature.get(link.get("from")) is not None
            and index_to_signature.get(link.get("to")) is not None
        }

        node_f1 = self._f1(gen_node_signatures, gt_node_signatures)
        edge_f1 = self._f1(gen_edge_signatures, gt_edge_signatures)
        return 0.5 * node_f1 + 0.5 * edge_f1

    def _node_signature(
        self, tool_name: str, arguments: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        normalized_args = []
        for arg in arguments:
            if isinstance(arg, str) and (
                arg.startswith("<node_") or arg.startswith("<REF_")
            ):
                normalized_args.append("<node>")
            else:
                normalized_args.append(str(arg))
        return tool_name, tuple(normalized_args)

    def _f1(self, pred: Set, gt: Set) -> float:
        if not pred and not gt:
            return 1.0
        if not pred or not gt:
            return 0.0
        overlap = len(pred & gt)
        precision = overlap / len(pred)
        recall = overlap / len(gt)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_latency_reward(
        self,
        generated_nodes: Dict[int, Dict],
        generated_edges: Set[Tuple[int, int]],
        baseline_latency_ms: Optional[float],
    ) -> float:
        if baseline_latency_ms is None or baseline_latency_ms <= 0:
            return 0.0

        node_latencies: Dict[int, float] = {}
        for idx, node in generated_nodes.items():
            resources = node["resources"]
            key = (
                node["tool_name"],
                node["input_size"],
                int(resources["cpu_core"]),
                float(resources["cpu_mem_gb"]),
                int(resources["gpu_sm"]),
                float(resources["gpu_mem_gb"]),
            )
            node_latencies[idx] = self.profiling_latency.get(
                key,
                node["registry_latency"],
            )

        makespan_ms = self._compute_makespan_ms(
            generated_nodes,
            generated_edges,
            node_latencies,
        )

        ratio = (baseline_latency_ms - makespan_ms) / (baseline_latency_ms + 1e-6)
        return max(-1.0, min(1.0, ratio))

    def _compute_makespan_ms(
        self,
        nodes: Dict[int, Dict],
        edges: Set[Tuple[int, int]],
        node_latencies: Dict[int, float],
    ) -> float:
        if not nodes:
            return 0.0

        parents: Dict[int, List[int]] = {idx: [] for idx in nodes}
        children: Dict[int, List[int]] = {idx: [] for idx in nodes}
        indegree: Dict[int, int] = {idx: 0 for idx in nodes}
        for src, dst in edges:
            if src not in nodes or dst not in nodes:
                continue
            parents[dst].append(src)
            children[src].append(dst)
            indegree[dst] += 1

        queue = [idx for idx, deg in indegree.items() if deg == 0]
        topo_order: List[int] = []
        while queue:
            cur = queue.pop()
            topo_order.append(cur)
            for dst in children[cur]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)

        if len(topo_order) != len(nodes):
            # Fallback guard: cycle should already be rejected in hard constraints.
            return sum(node_latencies.values())

        finish_time: Dict[int, float] = {}
        for idx in topo_order:
            start = 0.0
            if parents[idx]:
                start = max(finish_time[parent] for parent in parents[idx])
            finish_time[idx] = start + node_latencies.get(idx, 0.0)
        return max(finish_time.values())
