"""Standalone Hubbard dimer real-time dynamics with Qiskit built-ins.

This script is intentionally self-contained:
- No imports from local project modules.
- No edits to existing files are required.

Model:
- Two-site Fermi-Hubbard (dimer), half-filling.
- Jordan-Wigner mapping to qubits.

Time evolution:
- Approximate: built-in Suzuki-Trotter (or Lie-Trotter for order=1).
- Reference: built-in MatrixExponential synthesis.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

# Avoid matplotlib cache warnings triggered by qiskit_nature imports on some systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, MatrixExponential, SuzukiTrotter
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except Exception:
    pass


DEFAULT_NUM_SITES = 2  # Hubbard dimer
DEFAULT_ORDERING = "blocked"


@dataclass
class DynamicsPoint:
    """One time-point comparison between exact and Suzuki-Trotter evolution."""

    time: float
    fidelity: float
    state_l2_error: float
    energy_exact: float
    energy_trotter: float
    n_up_site0_exact: float
    n_up_site0_trotter: float
    n_dn_site0_exact: float
    n_dn_site0_trotter: float
    doublon_exact: float
    doublon_trotter: float


def _interleaved_to_blocked_permutation(num_sites: int) -> list[int]:
    """Map interleaved spin ordering (a1,b1,a2,b2,...) to blocked ordering (a1,a2,...,b1,b2,...)."""

    return [index for site in range(num_sites) for index in (site, num_sites + site)]


def _apply_spin_orbital_ordering(op, num_sites: int, ordering: str):
    """Return fermionic operator in the requested spin-orbital ordering."""

    normalized = ordering.strip().lower()
    if normalized == "interleaved":
        return op
    if normalized == "blocked":
        return op.permute_indices(_interleaved_to_blocked_permutation(num_sites))
    raise ValueError(f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'.")


def _build_qubit_hamiltonian(
    num_sites: int,
    hopping_t: float,
    onsite_u: float,
    boundary: str,
    ordering: str,
    mapper: JordanWignerMapper,
) -> SparsePauliOp:
    """Construct and JW-map the Fermi-Hubbard Hamiltonian."""

    boundary_condition = (
        BoundaryCondition.PERIODIC if boundary.strip().lower() == "periodic" else BoundaryCondition.OPEN
    )
    lattice = LineLattice(
        num_nodes=num_sites,
        edge_parameter=-hopping_t,
        onsite_parameter=0.0,
        boundary_condition=boundary_condition,
    )
    fermionic_op = FermiHubbardModel(lattice=lattice, onsite_interaction=onsite_u).second_q_op()
    fermionic_op = _apply_spin_orbital_ordering(fermionic_op, num_sites=num_sites, ordering=ordering)
    return mapper.map(fermionic_op).simplify(atol=1e-12)


def _half_filled_particle_numbers(num_sites: int) -> tuple[int, int]:
    """Half-filling particle tuple (n_alpha, n_beta)."""

    return ((num_sites + 1) // 2, num_sites // 2)


def _map_interleaved_index(index_interleaved: int, num_sites: int, ordering: str) -> int:
    """Convert interleaved spin-orbital index to target ordering index."""

    normalized = ordering.strip().lower()
    if normalized == "interleaved":
        return index_interleaved
    if normalized == "blocked":
        return _interleaved_to_blocked_permutation(num_sites)[index_interleaved]
    raise ValueError(f"Unsupported spin_orbital_ordering='{ordering}'. Use 'blocked' or 'interleaved'.")


def _build_initial_state_circuit(num_sites: int, ordering: str) -> QuantumCircuit:
    """Build a half-filled Slater determinant reference state circuit."""

    n_alpha, n_beta = _half_filled_particle_numbers(num_sites)
    occupied_interleaved: list[int] = []
    occupied_interleaved.extend(2 * site for site in range(n_alpha))  # alpha spin orbitals
    occupied_interleaved.extend(2 * site + 1 for site in range(n_beta))  # beta spin orbitals
    occupied = [_map_interleaved_index(i, num_sites=num_sites, ordering=ordering) for i in occupied_interleaved]

    num_qubits = 2 * num_sites
    qc = QuantumCircuit(num_qubits)
    for qubit in occupied:
        qc.x(qubit)
    return qc


def _number_operator_qubit(
    num_sites: int,
    site: int,
    spin: str,
    ordering: str,
    mapper: JordanWignerMapper,
) -> SparsePauliOp:
    """Build mapped number operator n_{site,spin}."""

    if spin not in {"up", "dn"}:
        raise ValueError("spin must be 'up' or 'dn'")
    orbital_interleaved = 2 * site + (0 if spin == "up" else 1)
    op = FermionicOp(
        {f"+_{orbital_interleaved} -_{orbital_interleaved}": 1.0},
        num_spin_orbitals=2 * num_sites,
    )
    op = _apply_spin_orbital_ordering(op, num_sites=num_sites, ordering=ordering)
    return mapper.map(op).simplify(atol=1e-12)


def _doublon_operator_qubit(
    num_sites: int,
    ordering: str,
    mapper: JordanWignerMapper,
) -> SparsePauliOp:
    """Build total doublon operator sum_i n_{i,up} n_{i,dn}."""

    total = SparsePauliOp.from_list([("I" * (2 * num_sites), 0.0)])
    for site in range(num_sites):
        up_interleaved = 2 * site
        dn_interleaved = 2 * site + 1
        op = FermionicOp(
            {f"+_{up_interleaved} -_{up_interleaved} +_{dn_interleaved} -_{dn_interleaved}": 1.0},
            num_spin_orbitals=2 * num_sites,
        )
        op = _apply_spin_orbital_ordering(op, num_sites=num_sites, ordering=ordering)
        total = (total + mapper.map(op)).simplify(atol=1e-12)
    return total


def _evolve_state(
    hamiltonian: SparsePauliOp | list[Pauli | SparsePauliOp],
    initial_state: QuantumCircuit,
    time: float,
    synthesis,
    decompose_reps: int = 0,
) -> Statevector:
    """Evolve a state by exp(-i * time * H) synthesized by the requested method.

    `decompose_reps` is used for product-formula paths so simulation applies the
    synthesized gate sequence rather than the exact matrix of PauliEvolutionGate.
    """

    qc = QuantumCircuit(initial_state.num_qubits)
    qc.compose(initial_state, inplace=True)
    qc.append(PauliEvolutionGate(operator=hamiltonian, time=time, synthesis=synthesis), range(initial_state.num_qubits))
    if decompose_reps > 0:
        qc = qc.decompose(reps=decompose_reps)
    return Statevector.from_instruction(qc)


def _split_hamiltonian_terms(hamiltonian: SparsePauliOp) -> list[SparsePauliOp]:
    """Split H into a list of single-Pauli terms for product-formula synthesis."""

    return [SparsePauliOp.from_list([(label, coeff)]) for label, coeff in hamiltonian.to_list()]


def _expectation_value(state: Statevector, operator: SparsePauliOp) -> float:
    """Real expectation value <state|operator|state>."""

    return float(np.real(state.expectation_value(operator)))


def _state_amplitudes_dict(state: Statevector, cutoff: float = 1e-10) -> dict[str, dict[str, float]]:
    """Serialize statevector amplitudes in computational basis q_(n-1)...q_0."""

    out: dict[str, dict[str, float]] = {}
    for basis_index, amp in enumerate(state.data):
        if abs(amp) < cutoff:
            continue
        bitstring = format(basis_index, f"0{state.num_qubits}b")
        out[bitstring] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _sparse_pauli_to_dict(op: SparsePauliOp, tol: float = 1e-12) -> dict[str, float | dict[str, float]]:
    """Serialize a SparsePauliOp into a stable dict label -> coefficient."""

    terms: dict[str, float | dict[str, float]] = {}
    for label, coeff in sorted(op.to_list(), key=lambda item: item[0]):
        c = complex(coeff)
        if abs(c) <= tol:
            continue
        if abs(c.imag) <= tol:
            terms[label] = float(c.real)
        else:
            terms[label] = {"re": float(c.real), "im": float(c.imag)}
    return terms


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Hubbard dimer real-time dynamics using Qiskit built-ins: "
            "Suzuki-Trotter (approximate) vs MatrixExponential (reference)."
        )
    )
    parser.add_argument("--hopping-t", type=float, default=1.0, help="Hopping parameter t.")
    parser.add_argument("--onsite-u", type=float, default=4.0, help="On-site interaction U.")
    parser.add_argument(
        "--boundary",
        choices=("open", "periodic"),
        default="periodic",
        help="Lattice boundary condition.",
    )
    parser.add_argument(
        "--ordering",
        choices=("blocked", "interleaved"),
        default=DEFAULT_ORDERING,
        help="Spin-orbital ordering before JW map.",
    )
    parser.add_argument("--t-final", type=float, default=2.0, help="Final evolution time.")
    parser.add_argument("--num-times", type=int, default=11, help="Number of time samples (includes t=0).")
    parser.add_argument(
        "--suzuki-order",
        type=int,
        default=2,
        help="Product-formula order. Use 1 for Lie-Trotter, or even number >=2 for Suzuki.",
    )
    parser.add_argument("--trotter-steps", type=int, default=8, help="Number of Trotter steps per sample time.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON path for machine-readable trajectory/results.",
    )
    parser.add_argument(
        "--dump-states",
        action="store_true",
        help="Include exact/trotter statevector amplitudes for every time point in JSON output.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if DEFAULT_NUM_SITES != 2:
        raise RuntimeError("This script is intended for the Hubbard dimer (num_sites=2).")
    if args.num_times < 2:
        raise ValueError("--num-times must be >= 2")
    if args.trotter_steps < 1:
        raise ValueError("--trotter-steps must be >= 1")
    if args.suzuki_order < 1:
        raise ValueError("--suzuki-order must be >= 1")
    if args.suzuki_order > 1 and args.suzuki_order % 2 != 0:
        raise ValueError("--suzuki-order must be 1 (Lie-Trotter) or an even integer (Suzuki).")

    mapper = JordanWignerMapper()
    hamiltonian = _build_qubit_hamiltonian(
        num_sites=DEFAULT_NUM_SITES,
        hopping_t=args.hopping_t,
        onsite_u=args.onsite_u,
        boundary=args.boundary,
        ordering=args.ordering,
        mapper=mapper,
    )
    hamiltonian_terms = _split_hamiltonian_terms(hamiltonian)
    initial_state = _build_initial_state_circuit(num_sites=DEFAULT_NUM_SITES, ordering=args.ordering)
    initial_statevector = Statevector.from_instruction(initial_state)

    n_up_site0 = _number_operator_qubit(
        num_sites=DEFAULT_NUM_SITES,
        site=0,
        spin="up",
        ordering=args.ordering,
        mapper=mapper,
    )
    n_dn_site0 = _number_operator_qubit(
        num_sites=DEFAULT_NUM_SITES,
        site=0,
        spin="dn",
        ordering=args.ordering,
        mapper=mapper,
    )
    doublon = _doublon_operator_qubit(
        num_sites=DEFAULT_NUM_SITES,
        ordering=args.ordering,
        mapper=mapper,
    )

    exact_synthesis = MatrixExponential()
    if args.suzuki_order == 1:
        trotter_synthesis = LieTrotter(reps=args.trotter_steps, preserve_order=True)
    else:
        trotter_synthesis = SuzukiTrotter(
            order=args.suzuki_order,
            reps=args.trotter_steps,
            preserve_order=True,
        )

    times = np.linspace(0.0, args.t_final, args.num_times)
    dynamics: list[DynamicsPoint] = []
    dynamics_json: list[dict[str, object]] = []

    for t in times:
        exact_state = _evolve_state(
            hamiltonian=hamiltonian,
            initial_state=initial_state,
            time=float(t),
            synthesis=exact_synthesis,
            decompose_reps=0,
        )
        trotter_state = _evolve_state(
            hamiltonian=hamiltonian_terms,
            initial_state=initial_state,
            time=float(t),
            synthesis=trotter_synthesis,
            decompose_reps=2,
        )

        overlap = np.vdot(exact_state.data, trotter_state.data)
        point = DynamicsPoint(
            time=float(t),
            fidelity=float(abs(overlap) ** 2),
            state_l2_error=float(np.linalg.norm(exact_state.data - trotter_state.data)),
            energy_exact=_expectation_value(exact_state, hamiltonian),
            energy_trotter=_expectation_value(trotter_state, hamiltonian),
            n_up_site0_exact=_expectation_value(exact_state, n_up_site0),
            n_up_site0_trotter=_expectation_value(trotter_state, n_up_site0),
            n_dn_site0_exact=_expectation_value(exact_state, n_dn_site0),
            n_dn_site0_trotter=_expectation_value(trotter_state, n_dn_site0),
            doublon_exact=_expectation_value(exact_state, doublon),
            doublon_trotter=_expectation_value(trotter_state, doublon),
        )
        dynamics.append(point)

        point_json = asdict(point)
        if args.dump_states:
            point_json["exact_state"] = _state_amplitudes_dict(exact_state)
            point_json["trotter_state"] = _state_amplitudes_dict(trotter_state)
        dynamics_json.append(point_json)

    print("Hubbard Dimer Real-Time Dynamics (Qiskit built-ins)")
    print(
        "Settings: "
        f"L=2, t={args.hopping_t}, U={args.onsite_u}, boundary={args.boundary}, "
        f"ordering={args.ordering}, t_final={args.t_final}, num_times={args.num_times}, "
        f"suzuki_order={args.suzuki_order}, trotter_steps={args.trotter_steps}"
    )
    print("Initial state (q3 q2 q1 q0):", initial_statevector.to_dict())
    print("-" * 120)
    print(
        "time    fidelity        ||dpsi||_2      <H>_exact       <H>_trotter     "
        "<n_up,0>_exact   <n_up,0>_trotter  doublon_exact   doublon_trotter"
    )
    for point in dynamics:
        print(
            f"{point.time:5.2f}  "
            f"{point.fidelity: .10f}  "
            f"{point.state_l2_error: .3e}  "
            f"{point.energy_exact: .10f}  "
            f"{point.energy_trotter: .10f}  "
            f"{point.n_up_site0_exact: .10f}  "
            f"{point.n_up_site0_trotter: .10f}  "
            f"{point.doublon_exact: .10f}  "
            f"{point.doublon_trotter: .10f}"
        )

    if args.output_json:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "description": (
                "Standalone Hubbard dimer time dynamics using Qiskit built-ins "
                "(Suzuki/Lie-Trotter vs MatrixExponential reference)."
            ),
            "settings": {
                "num_sites": DEFAULT_NUM_SITES,
                "hopping_t": args.hopping_t,
                "onsite_u": args.onsite_u,
                "boundary": args.boundary,
                "spin_orbital_ordering": args.ordering,
                "t_final": args.t_final,
                "num_times": args.num_times,
                "suzuki_order": args.suzuki_order,
                "trotter_steps": args.trotter_steps,
                "initial_state_label_qn_to_q0": list(initial_statevector.to_dict().keys())[0],
            },
            "hamiltonian_jw_terms": _sparse_pauli_to_dict(hamiltonian),
            "trajectory": dynamics_json,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
        print(f"Wrote JSON results to: {args.output_json}")


if __name__ == "__main__":
    main()
