"""Compile rtl/*.sv with tb/tb.sv and run every tb/programs/<name>.hex against <name>.expected.

Expected file: one "x<n> <hex8>" line per register (unlisted registers must be 0) and
"m<word index hex> <hex8>" lines for memory words (unlisted words must equal the program image).
Blank lines and lines starting with '#' are ignored. Usage: python run_tests.py [programs_dir]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IVERILOG = r"C:\iverilog\bin\iverilog.exe"
VVP = r"C:\iverilog\bin\vvp.exe"
MEM_WORDS = 16384
RUN_TIMEOUT_S = 300


def parse_image(path: Path) -> dict[int, int]:
    """Word index -> value of a $readmemh file (supports @addr and // comments)."""
    image: dict[int, int] = {}
    index = 0
    for line in path.read_text().splitlines():
        for token in line.split("//")[0].split():
            if token.startswith("@"):
                index = int(token[1:], 16)
            else:
                image[index] = int(token.replace("_", ""), 16)
                index += 1
    return image


def parse_state(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Registers and memory words listed in a dump or expected file."""
    regs: dict[int, int] = {}
    mem: dict[int, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, value = line.split()
        target = regs if name[0] == "x" else mem
        target[int(name[1:], 16 if name[0] == "m" else 10)] = int(value, 16)
    return regs, mem


def diff_state(hex_path: Path, expected_path: Path, dump_path: Path) -> list[str]:
    exp_regs, exp_mem = parse_state(expected_path)
    image = parse_image(hex_path)
    got_regs, got_mem = parse_state(dump_path)
    problems = []
    for n in range(32):
        want, got = exp_regs.get(n, 0), got_regs[n]
        if want != got:
            problems.append(f"x{n}: expected {want:08X}, got {got:08X}")
    for i in range(MEM_WORDS):
        want = exp_mem.get(i, image.get(i, 0))
        if want != got_mem[i]:
            problems.append(f"mem[{i:04X}] (addr {4 * i:05X}): expected {want:08X}, got {got_mem[i]:08X}")
    return problems


def run_test(sim: Path, hex_path: Path, workdir: Path) -> list[str]:
    expected_path = hex_path.with_suffix(".expected")
    if not expected_path.exists():
        return [f"missing {expected_path.name}"]
    dump_path = workdir / f"{hex_path.stem}.dump"
    try:
        run = subprocess.run(
            [VVP, str(sim), f"+hex={hex_path.as_posix()}", f"+out={dump_path.as_posix()}"],
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return [f"simulator ran longer than {RUN_TIMEOUT_S} s"]
    errors = [l for l in run.stdout.splitlines() if "[tb::error]" in l]
    if errors or run.returncode != 0 or not dump_path.exists():
        return errors or [f"simulator failed (exit {run.returncode}): {run.stderr.strip()}"]
    return diff_state(hex_path, expected_path, dump_path)


def main() -> int:
    programs = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "tb" / "programs"
    sources = sorted((ROOT / "rtl").glob("*.sv"))
    if not sources:
        print("[run_tests::error] no .sv files in rtl/")
        return 1
    tests = sorted(programs.glob("*.hex"))
    if not tests:
        print(f"[run_tests::error] no .hex programs in {programs}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        sim = workdir / "sim.vvp"
        build = subprocess.run(
            [IVERILOG, "-g2012", "-s", "tb", "-o", str(sim), *map(str, sources), str(ROOT / "tb" / "tb.sv")],
            capture_output=True, text=True,
        )
        if build.returncode != 0:
            print(f"[run_tests::error] compile failed\n{build.stdout}{build.stderr}")
            return 1

        passed = 0
        for hex_path in tests:
            problems = run_test(sim, hex_path, workdir)
            if problems:
                print(f"FAIL {hex_path.stem}")
                for p in problems[:10]:
                    print(f"     {p}")
                if len(problems) > 10:
                    print(f"     ... {len(problems) - 10} more")
            else:
                print(f"PASS {hex_path.stem}")
                passed += 1
    print(f"{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
