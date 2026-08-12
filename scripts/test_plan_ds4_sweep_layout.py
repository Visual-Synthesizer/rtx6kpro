from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).with_name("plan-ds4-sweep-layout.py")
SPEC = spec_from_file_location("ds4_sweep_layout", SCRIPT)
assert SPEC and SPEC.loader
LAYOUT = module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)


def test_assignment_is_independent_of_completed_cases() -> None:
    rows = LAYOUT.build_layout(
        tp=2,
        backends=("a16", "a8"),
        modes=("mtp0", "k5", "k7", "k7-dynamic"),
        gpu_groups=("0,1", "2,3", "4,5", "6,7"),
        port_base=5000,
    )
    assignments = {(row[1], row[2]): row[4:] for row in rows}

    pending = [row for row in rows if row[2] not in {"k5", "k7"}]

    assert {(row[1], row[2]): row[4:] for row in pending} == {
        key: value for key, value in assignments.items() if key[1] not in {"k5", "k7"}
    }
    assert assignments[("a16", "k7-dynamic")] == (0, 3, "6,7", 5203)
    assert assignments[("a8", "mtp0")] == (1, 0, "0,1", 5220)


def test_rejects_duplicate_gpu_groups() -> None:
    try:
        LAYOUT.build_layout(
            tp=4,
            backends=("a16",),
            modes=("mtp0",),
            gpu_groups=("0,1,2,3", "0,1,2,3"),
            port_base=5000,
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate GPU groups were accepted")
