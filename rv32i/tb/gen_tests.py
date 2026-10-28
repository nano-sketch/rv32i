"""Generate the RV32I test programs into programs/: <name>.hex (program image) and <name>.expected (final state).

Each program carries values worked out by hand (register and memory expectations written in this file,
computed with plain Python formulas, not with refmodel.py). The generator assembles the program, runs it
on refmodel.py, and refuses to write anything unless the model agrees with every hand expectation.
Usage: python gen_tests.py
"""
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from asm import I_OPS, LOAD_OPS, R_OPS, SHIFT_OPS, STORE_OPS, assemble
from refmodel import MASK, MEM_BYTES, run

OUT_DIR = Path(__file__).resolve().parent / "programs"
DATA = 0x4000  # data area base, above every program
MIN_INT = 0x80000000
MAX_INT = 0x7FFFFFFF

ENCODING_CHECKS = {  # hand-encoded from the RISC-V unprivileged spec
    "addi x1, x0, 1": 0x00100093,
    "addi x1, x0, -1": 0xFFF00093,
    "add x3, x1, x2": 0x002081B3,
    "sub x3, x1, x2": 0x402081B3,
    "lw x5, 4(x6)": 0x00432283,
    "sw x5, 8(x6)": 0x00532423,
    "sb x5, -1(x6)": 0xFE530FA3,
    "beq x1, x2, 8": 0x00208463,
    "beq x0, x0, -4": 0xFE000EE3,
    "bgeu x1, x2, 16": 0x0020F863,
    "jal x1, 8": 0x008000EF,
    "jal x0, -4": 0xFFDFF06F,
    "jalr x1, 4(x2)": 0x004100E7,
    "lui x5, 0x12345": 0x123452B7,
    "lui x1, 0xFFFFF": 0xFFFFF0B7,
    "auipc x2, 1": 0x00001117,
    "srai x1, x2, 3": 0x40315093,
    "sltiu x1, x2, -1": 0xFFF13093,
    "ecall": 0x00000073,
    "fence": 0x0FF0000F,
    "nop": 0x00000013,
    "ret": 0x00008067,
}


@dataclass
class Program:
    name: str
    source: str
    regs: dict[int, int] = field(default_factory=dict)  # hand-computed registers (others are not checked)
    mem: dict[int, int] = field(default_factory=dict)  # hand-computed words, keyed by byte address


def check_assembler() -> None:
    for text, want in ENCODING_CHECKS.items():
        got = assemble(text)
        assert got == [want], f"[gen_tests::error] '{text}' assembled to {got[0]:08X}, expected {want:08X}"
    assert assemble("li x1, 0x89ABCDEF") == [0x89ABD0B7, 0xDEF08093]
    assert assemble("top: j top\nbeqz x0, top") == [0x0000006F, 0xFE000EE3]


def check_model(program: Program, regs: list[int], mem: list[int]) -> None:
    for n, want in program.regs.items():
        assert regs[n] == want & MASK, f"[gen_tests::error] {program.name}: x{n} model {regs[n]:08X}, hand {want & MASK:08X}"
    for addr, want in program.mem.items():
        assert mem[addr >> 2] == want & MASK, (
            f"[gen_tests::error] {program.name}: mem[{addr:#X}] model {mem[addr >> 2]:08X}, hand {want & MASK:08X}"
        )


def write_program(program: Program) -> None:
    words = assemble(program.source)
    assert len(words) * 4 <= DATA, f"[gen_tests::error] {program.name}: program overlaps the data area"
    machine = run(words)
    check_model(program, machine.regs, machine.mem)

    (OUT_DIR / f"{program.name}.hex").write_text("".join(f"{w:08X}\n" for w in words))
    image = words + [0] * (len(machine.mem) - len(words))
    lines = [f"x{n} {v:08X}" for n, v in enumerate(machine.regs)]
    lines += [f"m{i:04X} {v:08X}" for i, (v, old) in enumerate(zip(machine.mem, image)) if v != old]
    (OUT_DIR / f"{program.name}.expected").write_text("\n".join(lines) + "\n")


# ---- helpers for hand expectations -------------------------------------------------------------

def signed(value: int) -> int:
    return value - (1 << 32) if value & MIN_INT else value


def sll(value: int, n: int) -> int:
    return (value << n) & MASK


def srl(value: int, n: int) -> int:
    return value >> n


def sra(value: int, n: int) -> int:
    return (signed(value) >> n) & MASK


# ---- directed programs -------------------------------------------------------------------------

def alu_basic() -> Program:
    src = """
        li x1, 0x7FFFFFFF
        li x2, 1
        add x3, x1, x2            # overflow wraps to 0x80000000
        sub x4, x3, x2            # and back
        li x5, -1
        add x6, x5, x2            # 0xFFFFFFFF + 1 = 0
        sub x7, x0, x2            # 0 - 1
        sub x8, x6, x2
        li x9, 0xF0F0F0F0
        li x10, 0x0FF00FF0
        and x11, x9, x10
        or x12, x9, x10
        xor x13, x9, x10
        andi x14, x9, 0x7FF
        ori x15, x10, -1          # immediate is sign-extended
        xori x16, x9, -1          # bitwise not
        xori x17, x9, 0xFF
        addi x18, x1, -2048
        addi x19, x1, 2047
        add x20, x3, x3           # 0x80000000 * 2 wraps to 0
        sub x21, x3, x1           # 0x80000000 - 0x7FFFFFFF = 1
        sub x22, x2, x3           # 1 - 0x80000000 = 0x80000001
        addi x23, x3, -1
        andi x24, x5, -2048       # 0xFFFFF800
        ecall
    """
    regs = {
        1: MAX_INT, 2: 1, 3: MIN_INT, 4: MAX_INT, 5: MASK, 6: 0, 7: MASK, 8: MASK,
        9: 0xF0F0F0F0, 10: 0x0FF00FF0, 11: 0x00F000F0, 12: 0xFFF0FFF0, 13: 0xFF00FF00,
        14: 0x0F0, 15: MASK, 16: 0x0F0F0F0F, 17: 0xF0F0F00F, 18: 0x7FFFF7FF, 19: 0x800007FE,
        20: 0, 21: 1, 22: 0x80000001, 23: MAX_INT, 24: 0xFFFFF800,
    }
    return Program("alu_basic", src, regs)


def upper_immediates() -> Program:
    src = """
        auipc x1, 0               # pc 0x00
        auipc x2, 1               # pc 0x04
        auipc x3, 0xFFFFF         # pc 0x08
        lui x4, 0x12345           # pc 0x0C
        addi x4, x4, 0x678
        li x5, 0x89ABCDEF         # needs the carry correction for the negative low part
        auipc x6, 0x80000         # pc 0x1C
        lui x7, 0
        lui x0, 5
        auipc x0, 5
        lui x8, 0xFFFFF
        lui x9, 0x80000
        li x10, 0x7FFFF800
        li x11, 0xFFFFF800
        ecall
    """
    regs = {
        0: 0, 1: 0, 2: 0x1004, 3: 0xFFFFF008, 4: 0x12345678, 5: 0x89ABCDEF, 6: 0x8000001C, 7: 0,
        8: 0xFFFFF000, 9: MIN_INT, 10: 0x7FFFF800, 11: 0xFFFFF800,
    }
    return Program("upper_immediates", src, regs)


def slt_edges() -> Program:
    src = """
        li x1, 0x80000000         # most negative
        li x2, 0x7FFFFFFF         # most positive
        li x3, -1
        li x5, 1
        slt x10, x1, x2           # 1
        slt x11, x2, x1           # 0
        sltu x12, x1, x2          # 0
        sltu x13, x2, x1          # 1
        slt x14, x3, x0           # 1
        sltu x15, x3, x0          # 0
        sltu x16, x0, x3          # 1
        slt x17, x1, x1           # 0
        sltu x18, x1, x1          # 0
        slt x19, x3, x5           # 1
        sltu x20, x5, x3          # 1
        slt x21, x0, x3           # 0
        slti x22, x3, 0           # 1
        sltiu x23, x3, 0          # 0
        sltiu x24, x0, -1         # immediate sign-extends to 0xFFFFFFFF: 1
        slti x25, x1, -2048       # 1
        slti x26, x0, -2048       # 0
        sltiu x27, x0, 1          # 1
        sltiu x28, x5, 1          # 0
        slti x29, x2, 2047        # 0
        slti x30, x0, 2047        # 1
        sltiu x31, x1, -1         # 1
        ecall
    """
    want = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    regs = {10 + i: v for i, v in enumerate(want)}
    return Program("slt_edges", src, regs)


def shifts() -> Program:
    lines = ["li x31, 0x4000", "li x1, 0x80000001", "li x3, 0x7EDCBA98"]
    mem: dict[int, int] = {}
    slot = 0
    for reg, value in (("x1", 0x80000001), ("x3", 0x7EDCBA98)):
        for n in range(32):
            # rs2 keeps junk above bit 4: only its low 5 bits may count
            lines.append(f"li x2, {-32 + n if n % 2 == 0 else 0x7A0 + n}")
            for op, fn in (("slli", sll), ("srli", srl), ("srai", sra)):
                lines += [f"{op} x4, {reg}, {n}", f"sw x4, {4 * slot}(x31)"]
                mem[DATA + 4 * slot] = fn(value, n)
                slot += 1
            for op, fn in (("sll", sll), ("srl", srl), ("sra", sra)):
                lines += [f"{op} x4, {reg}, x2", f"sw x4, {4 * slot}(x31)"]
                mem[DATA + 4 * slot] = fn(value, n)
                slot += 1
    lines.append("ecall")
    return Program("shifts", "\n".join(lines), {1: 0x80000001, 3: 0x7EDCBA98, 4: 0, 31: DATA}, mem)


BRANCH_CASES = [  # (op, rs1 value, rs2 value, taken)
    ("beq", 5, 5, True), ("beq", 5, 6, False), ("beq", -1, -1, True), ("beq", 0, MIN_INT, False),
    ("bne", 5, 6, True), ("bne", 7, 7, False), ("bne", 0, MIN_INT, True), ("bne", -1, -1, False),
    ("blt", -1, 1, True), ("blt", MIN_INT, MAX_INT, True), ("blt", 1, -1, False), ("blt", 3, 3, False),
    ("blt", MAX_INT, MIN_INT, False), ("blt", -5, -4, True),
    ("bge", 1, -1, True), ("bge", 3, 3, True), ("bge", -1, 1, False), ("bge", MAX_INT, MIN_INT, True),
    ("bge", MIN_INT, MAX_INT, False), ("bge", -4, -5, True),
    ("bltu", 1, -1, True), ("bltu", -1, 1, False), ("bltu", 3, 3, False), ("bltu", MAX_INT, MIN_INT, True),
    ("bltu", MIN_INT, MAX_INT, False), ("bltu", 0, 1, True),
    ("bgeu", -1, 1, True), ("bgeu", 3, 3, True), ("bgeu", 1, -1, False), ("bgeu", MIN_INT, MAX_INT, True),
    ("bgeu", MAX_INT, MIN_INT, False), ("bgeu", 0, 0, True),
]


def branch_block(i: int, op: str, a: int, b: int, taken: bool, layout: str) -> list[str]:
    """x20 counts blocks that behaved correctly; x21 counts blocks that did not."""
    head = [f"li x1, {a}", f"li x2, {b}"]
    if layout == "forward" and taken:
        body = [f"{op} x1, x2, t{i}", "addi x21, x21, 1", f"t{i}:", "addi x20, x20, 1"]
    elif layout == "forward":
        body = [f"{op} x1, x2, t{i}", "addi x20, x20, 1", f"jal x0, e{i}", f"t{i}:", "addi x21, x21, 1", f"e{i}:"]
    elif taken:
        body = [f"jal x0, f{i}", f"b{i}:", "addi x20, x20, 1", f"jal x0, e{i}", f"f{i}:",
                f"{op} x1, x2, b{i}", "addi x21, x21, 1", f"e{i}:"]
    else:
        body = [f"jal x0, f{i}", f"b{i}:", "addi x21, x21, 1", f"jal x0, e{i}", f"f{i}:",
                f"{op} x1, x2, b{i}", "addi x20, x20, 1", f"e{i}:"]
    return head + body


def branches() -> Program:
    lines: list[str] = []
    count = 0
    for layout in ("forward", "backward"):
        for op, a, b, taken in BRANCH_CASES:
            lines += branch_block(count, op, a, b, taken, layout)
            count += 1
    lines.append("ecall")
    return Program("branches", "\n".join(lines), {20: count, 21: 0})


def branch_far() -> Program:
    """Branches near the +-4 KiB limit of the B-type offset, taken forward and backward."""
    filler = ["nop"] * 1000
    lines = (
        ["beq x0, x0, far", "addi x21, x21, 1"] + filler + ["far:", "addi x20, x20, 1",
         "jal x0, code"] + ["target:", "addi x20, x20, 10", "jal x0, end"] + filler
        + ["code:", "bge x0, x0, target", "addi x21, x21, 1", "end:", "bne x0, x0, far", "ecall"]
    )
    return Program("branch_far", "\n".join(lines), {20: 11, 21: 0})


def jumps() -> Program:
    src = """
        jal ra, func              # 0x00: ra = 4
        addi x10, x10, 1          # 0x04: after return, x10 = 40 + 1
        jal x5, over              # 0x08: x5 = 0x0C
        addi x11, x0, 99          # 0x0C: skipped
    over:
        auipc x6, 0               # 0x10
        addi x6, x6, 16           # 0x14: x6 = 0x20
        jalr x7, 0(x6)            # 0x18: x7 = 0x1C, jumps to 0x20
        addi x12, x0, 99          # 0x1C: skipped
        auipc x9, 0               # 0x20
        addi x9, x9, 0x2D         # 0x24: x9 = 0x4D, odd
        jalr x8, 0(x9)            # 0x28: x8 = 0x2C, target 0x4C (bit 0 cleared)
        addi x13, x0, 99          # 0x2C..0x48: skipped
        addi x13, x0, 99
        addi x13, x0, 99
        addi x13, x0, 99
        addi x13, x0, 99
        addi x13, x0, 99
        addi x13, x0, 99
        addi x13, x0, 99
        auipc x14, 0              # 0x4C
        addi x14, x14, 16         # 0x50: x14 = 0x5C
        jalr x14, 0(x14)          # 0x54: rd == rs1: x14 = 0x58, target 0x5C
        addi x15, x0, 99          # 0x58: skipped
        auipc x16, 0              # 0x5C
        addi x16, x16, 32         # 0x60: x16 = 0x7C
        jalr x17, -8(x16)         # 0x64: x17 = 0x68, target 0x74
        addi x18, x0, 99          # 0x68, 0x6C, 0x70: skipped
        addi x18, x0, 99
        addi x18, x0, 99
        jal x0, done              # 0x74
    func:
        addi x10, x0, 40          # 0x78
        ret                       # 0x7C
    done:
        ecall                     # 0x80
    """
    regs = {
        1: 4, 5: 0x0C, 6: 0x20, 7: 0x1C, 8: 0x2C, 9: 0x4D, 10: 41, 11: 0, 12: 0, 13: 0,
        14: 0x58, 15: 0, 16: 0x7C, 17: 0x68, 18: 0,
    }
    return Program("jumps", src, regs)


def loads() -> Program:
    src = """
        li x1, 0x4000
        li x2, 0x8081F27F
        sw x2, 0(x1)
        li x2, 0x7F0180FF
        sw x2, 4(x1)
        lb x10, 0(x1)
        lb x11, 1(x1)
        lb x12, 2(x1)
        lb x13, 3(x1)
        lbu x14, 0(x1)
        lbu x15, 1(x1)
        lbu x16, 2(x1)
        lbu x17, 3(x1)
        lh x18, 0(x1)
        lh x19, 2(x1)
        lhu x20, 0(x1)
        lhu x21, 2(x1)
        lw x22, 0(x1)
        lh x23, 6(x1)
        lh x24, 4(x1)
        lb x25, 4(x1)
        lbu x26, 4(x1)
        lb x27, 7(x1)
        addi x2, x1, 8
        lw x28, -4(x2)            # negative offset
        lhu x29, -2(x2)
        lw x0, 0(x1)              # load into x0 is discarded
        lb x30, -8(x2)
        ecall
    """
    regs = {
        10: 0x7F, 11: 0xFFFFFFF2, 12: 0xFFFFFF81, 13: 0xFFFFFF80, 14: 0x7F, 15: 0xF2, 16: 0x81, 17: 0x80,
        18: 0xFFFFF27F, 19: 0xFFFF8081, 20: 0xF27F, 21: 0x8081, 22: 0x8081F27F, 23: 0x7F01, 24: 0xFFFF80FF,
        25: MASK, 26: 0xFF, 27: 0x7F, 28: 0x7F0180FF, 29: 0x7F01, 30: 0x7F, 0: 0,
    }
    return Program("loads", src, regs, {DATA: 0x8081F27F, DATA + 4: 0x7F0180FF})


def stores() -> Program:
    fill = "\n".join(f"        sw x3, {4 * i}(x1)" for i in range(12))
    src = f"""
        li x1, 0x4000
        li x2, 0x11223344
        li x3, 0xAABBCCDD
        li x4, 0x8000FFFE
{fill}
        sb x2, 0(x1)              # word 0: byte 0
        sb x2, 5(x1)              # word 1: byte 1
        sb x2, 10(x1)             # word 2: byte 2
        sb x2, 15(x1)             # word 3: byte 3
        sh x2, 16(x1)             # word 4: low half
        sh x2, 22(x1)             # word 5: high half
        sw x2, 24(x1)             # word 6: whole word
        addi x5, x1, 0x20
        sb x2, -3(x5)             # word 7: byte 1 via negative offset
        sh x4, 34(x1)             # word 8: high half = 0xFFFE
        sb x4, 39(x1)             # word 9: byte 3 = 0xFE
        sw x0, 40(x1)             # word 10: x0 stores zero
        sb x0, 45(x1)             # word 11: byte 1 cleared
        sw x2, 1536(x0)           # absolute address through x0
        ecall
    """
    words = [0xAABBCC44, 0xAABB44DD, 0xAA44CCDD, 0x44BBCCDD, 0xAABB3344, 0x3344CCDD, 0x11223344,
             0xAABB44DD, 0xFFFECCDD, 0xFEBBCCDD, 0, 0xAABB00DD]
    mem = {DATA + 4 * i: w for i, w in enumerate(words)} | {1536: 0x11223344}
    return Program("stores", src, {1: DATA, 2: 0x11223344, 3: 0xAABBCCDD, 4: 0x8000FFFE, 5: DATA + 0x20}, mem)


def x0_behaviour() -> Program:
    src = """
        li x1, 7
        li x9, 0x4000
        li x8, 0x1234
        sw x8, 0(x9)
        addi x0, x0, 5
        add x0, x1, x1
        lui x0, 1
        auipc x0, 1
        lw x0, 0(x9)
        sub x0, x0, x1
        slli x0, x1, 3
        jal x0, next
    next:
        add x2, x0, x1            # x0 reads as 0
        sub x6, x0, x1
        or x7, x0, x1
        sltu x10, x0, x1
        slt x11, x0, x1
        add x12, x0, x0
        beq x0, x0, taken
        addi x13, x0, 99
    taken:
        bne x0, x0, bad
        sw x0, 4(x9)
        addi x14, x0, 1
        jal x0, fin
    bad:
        addi x14, x0, 99
    fin:
        ecall
    """
    return Program(
        "x0_behaviour", src,
        {0: 0, 1: 7, 2: 7, 6: 0xFFFFFFF9, 7: 7, 10: 1, 11: 1, 12: 0, 13: 0, 14: 1, 8: 0x1234},
        {DATA: 0x1234, DATA + 4: 0},
    )


def hazards() -> Program:
    src = """
        li x1, 0x4000
        addi x2, x0, 5
        add x3, x2, x2            # back-to-back dependencies
        add x4, x3, x2
        add x5, x4, x3
        sw x5, 0(x1)
        lw x6, 0(x1)
        addi x7, x6, 1            # load-use
        sw x7, 4(x1)
        lw x8, 4(x1)
        sw x8, 8(x1)              # store the value just loaded
        lw x9, 8(x1)
        beq x9, x7, ok            # branch on a just-loaded value
        addi x10, x0, 99
    ok:
        addi x10, x10, 1
        addi x11, x1, 12
        sw x11, 12(x1)
        lw x12, 12(x1)
        lw x13, 0(x12)            # load through a just-loaded pointer
        auipc x14, 0
        addi x14, x14, 16
        jalr x0, 0(x14)           # jump through a just-computed address
        addi x17, x0, 99
        addi x15, x0, 1           # write after write
        addi x15, x0, 2
        add x15, x15, x15
        addi x16, x16, 1          # rd == rs1 chain
        addi x16, x16, 1
        addi x16, x16, 1
        addi x16, x16, 1
        li x18, 0x44332211
        sw x18, 16(x1)
        sb x2, 17(x1)             # sb then lw on the same word
        lw x19, 16(x1)
        sh x2, 18(x1)
        lw x20, 16(x1)
        lbu x21, 17(x1)
        beq x21, x2, skip         # taken branch: next instruction must not write
        addi x22, x0, 99
    skip:
        sw x7, 20(x1)
        ecall
    """
    regs = {
        2: 5, 3: 10, 4: 15, 5: 25, 6: 25, 7: 26, 8: 26, 9: 26, 10: 1, 11: DATA + 12, 12: DATA + 12, 13: DATA + 12,
        15: 4, 16: 4, 17: 0, 18: 0x44332211, 19: 0x44330511, 20: 0x00050511, 21: 5, 22: 0,
    }
    mem = {DATA: 25, DATA + 4: 26, DATA + 8: 26, DATA + 12: DATA + 12, DATA + 16: 0x00050511, DATA + 20: 26}
    return Program("hazards", src, regs, mem)


def halt_stops() -> list[Program]:
    after = """
        addi x1, x0, 1
        sw x1, 1536(x0)
        ecall
        addi x1, x0, 2            # nothing after ECALL may take effect
        sw x1, 1540(x0)
        addi x2, x0, 3
        jal x3, 8
        sw x1, 1536(x0)
    """
    first = "ecall\naddi x1, x0, 7\nsw x1, 1536(x0)"
    store_before = "li x1, 0x4000\nli x2, 0x5A5A5A5A\nsw x2, 0(x1)\necall\nsw x0, 0(x1)\naddi x2, x0, 0"
    load_before = "li x1, 0x4000\nsw x1, 0(x1)\nlw x2, 0(x1)\necall\naddi x2, x0, 1"
    branch_to = "beq x0, x0, end\naddi x1, x0, 1\nend:\necall\naddi x1, x0, 2"
    return [
        Program("halt_after_ecall", after, {1: 1, 2: 0, 3: 0}, {1536: 1, 1540: 0}),
        Program("halt_first", first, {1: 0}, {1536: 0}),
        Program("halt_after_store", store_before, {1: DATA, 2: 0x5A5A5A5A}, {DATA: 0x5A5A5A5A}),
        Program("halt_after_load", load_before, {1: DATA, 2: DATA}, {DATA: DATA}),
        Program("halt_after_branch", branch_to, {1: 0}),
    ]


def fence_nop() -> Program:
    src = "addi x1, x0, 1\nfence\naddi x1, x1, 1\nfence\nfence\nlui x2, 1\nnop\nnop\necall"
    return Program("fence_nop", src, {1: 2, 2: 0x1000})


def memory_edges() -> Program:
    src = """
        lw x5, 0(x0)             # reads the first instruction (this very lw)
        li x1, 0xFFFC
        li x2, 0xDEADBEEF
        sw x2, 0(x1)
        lw x3, 0(x1)
        lb x4, 3(x1)
        lbu x6, 3(x1)
        lh x7, 2(x1)
        lhu x8, 2(x1)
        sb x0, 3(x1)
        sh x0, 0(x1)
        lw x9, 0(x1)
        li x10, 0x10000
        lw x11, -4(x10)
        sb x2, -1(x10)           # last byte of memory
        lw x12, -4(x10)
        sh x2, -2(x10)
        lw x13, -4(x10)
        ecall
    """
    regs = {
        5: 0x00002283, 1: 0xFFFC, 2: 0xDEADBEEF, 3: 0xDEADBEEF, 4: 0xFFFFFFDE, 6: 0xDE, 7: 0xFFFFDEAD,
        8: 0xDEAD, 9: 0x00AD0000, 10: 0x10000, 11: 0x00AD0000, 12: 0xEFAD0000, 13: 0xBEEF0000,
    }
    return Program("memory_edges", src, regs, {0xFFFC: 0xBEEF0000})


# ---- loops and recursion ------------------------------------------------------------------------

def mul_block(a: int, b: int, ra: str, rb: str, rr: str, tag: str) -> list[str]:
    """Shift-and-add multiplication: rr = a * b mod 2^32."""
    return [
        f"li {ra}, {a}", f"li {rb}, {b}", f"li {rr}, 0",
        f"m{tag}:", "andi x4, " + rb + ", 1", f"beqz x4, s{tag}", f"add {rr}, {rr}, {ra}",
        f"s{tag}:", f"slli {ra}, {ra}, 1", f"srli {rb}, {rb}, 1", f"bnez {rb}, m{tag}",
    ]


def loop_programs() -> list[Program]:
    total = Program(
        "loop_sum",
        "li x10, 0\nli x11, 1\nli x12, 100\nloop:\nadd x10, x10, x11\naddi x11, x11, 1\nbge x12, x11, loop\necall",
        {10: 5050, 11: 101},
    )
    count = Program(
        "loop_count",
        "li x1, 3000\nloop:\naddi x1, x1, -1\naddi x2, x2, 3\nbnez x1, loop\necall",
        {1: 0, 2: 9000},
    )
    fibs = [0, 1]
    while len(fibs) < 27:
        fibs.append(fibs[-1] + fibs[-2])
    fib = Program(
        "loop_fib",
        """
            li x31, 0x4000
            li x5, 0
            li x6, 1
            li x7, 0
            li x8, 25
        loop:
            sw x5, 0(x31)
            add x9, x5, x6
            mv x5, x6
            mv x6, x9
            addi x31, x31, 4
            addi x7, x7, 1
            blt x7, x8, loop
            ecall
        """,
        {5: fibs[25], 6: fibs[26], 7: 25, 31: DATA + 100},
        {DATA + 4 * i: fibs[i] for i in range(25)},
    )
    mul_src = "\n".join(
        mul_block(12345, 6789, "x1", "x2", "x3", "a") + mul_block(0xDEADBEEF, 0x12345678, "x5", "x6", "x7", "b") + ["ecall"]
    )
    mul = Program("loop_mul", mul_src, {3: 83810205, 7: (0xDEADBEEF * 0x12345678) & MASK, 2: 0, 6: 0})
    gcd = Program(
        "loop_gcd",
        """
            li x1, 1071
            li x2, 462
        loop:
            beq x1, x2, done
            blt x1, x2, less
            sub x1, x1, x2
            j loop
        less:
            sub x2, x2, x1
            j loop
        done:
            ecall
        """,
        {1: 21, 2: 21},
    )
    values = [5, -3, MAX_INT, MIN_INT, 0, -1, 100, 5]
    init = "\n".join(f"li x2, {v}\nsw x2, {4 * i}(x1)" for i, v in enumerate(values))
    sort = Program(
        "loop_sort",
        f"""
            li x1, 0x4000
{init}
            li x3, 8
        outer:
            addi x3, x3, -1
            beqz x3, done
            mv x4, x0
            mv x5, x1
        inner:
            lw x6, 0(x5)
            lw x7, 4(x5)
            bge x7, x6, noswap
            sw x7, 0(x5)
            sw x6, 4(x5)
        noswap:
            addi x5, x5, 4
            addi x4, x4, 1
            blt x4, x3, inner
            j outer
        done:
            ecall
        """,
        {3: 0, 1: DATA},
        {DATA + 4 * i: v for i, v in enumerate(sorted((v & MASK for v in values), key=signed))},
    )
    return [total, count, fib, mul, gcd, sort]


def recursion() -> Program:
    src = """
        li sp, 0x8000
        li a0, 12
        call fact
        mv s1, a0                  # 12! = 479001600
        li a0, 15
        call fib
        mv s2, a0                  # fib(15) = 610
        li a0, 200
        call sumto
        mv s3, a0                  # 200 + ... + 1 = 20100
        ecall
    fact:
        addi sp, sp, -8
        sw ra, 4(sp)
        sw a0, 0(sp)
        li t0, 2
        blt a0, t0, fact_base
        addi a0, a0, -1
        call fact
        lw t1, 0(sp)               # multiply a0 by t1 with repeated addition
        li t2, 0
    fact_mul:
        add t2, t2, a0
        addi t1, t1, -1
        bnez t1, fact_mul
        mv a0, t2
        j fact_ret
    fact_base:
        li a0, 1
    fact_ret:
        lw ra, 4(sp)
        addi sp, sp, 8
        ret
    fib:
        li t0, 2
        blt a0, t0, fib_base
        addi sp, sp, -12
        sw ra, 8(sp)
        sw a0, 4(sp)
        addi a0, a0, -1
        call fib
        sw a0, 0(sp)
        lw a0, 4(sp)
        addi a0, a0, -2
        call fib
        lw t1, 0(sp)
        add a0, a0, t1
        lw ra, 8(sp)
        addi sp, sp, 12
    fib_base:
        ret
    sumto:
        beqz a0, sum_base
        addi sp, sp, -8
        sw ra, 4(sp)
        sw a0, 0(sp)
        addi a0, a0, -1
        call sumto
        lw t0, 0(sp)
        add a0, a0, t0
        lw ra, 4(sp)
        addi sp, sp, 8
    sum_base:
        ret
    """
    return Program("recursion", src, {2: 0x8000, 9: 479001600, 18: 610, 19: 20100})


# ---- random programs ------------------------------------------------------------------------------

EDGE_VALUES = [0, 1, MASK, MAX_INT, MIN_INT, 0x80000001, 0xFFFF, 0x10000, 0xFF, 0x80, 0x8000, 0x7FFF]
IMM_EDGES = [-2048, -1, 0, 1, 2047, 31, -32]
DEST = range(16)  # x16 is the loop counter and x31 the data base: never random destinations


class RandomProgram:
    def __init__(self, seed: int, items: int, weights: dict[str, int]) -> None:
        self.rng = random.Random(seed)
        self.labels = 0
        self.lines: list[str] = []
        self.build(items, weights)

    def value(self) -> int:
        return self.rng.choice(EDGE_VALUES) if self.rng.random() < 0.4 else self.rng.getrandbits(32)

    def imm(self) -> int:
        return self.rng.choice(IMM_EDGES) if self.rng.random() < 0.3 else self.rng.randrange(-2048, 2048)

    def reg(self) -> int:
        return self.rng.choice(DEST)

    def label(self) -> str:
        self.labels += 1
        return f"L{self.labels}"

    def alu(self) -> str:
        kind = self.rng.choice(["r", "r", "i", "shift", "lui", "auipc"])
        rd, rs1, rs2 = self.reg(), self.reg(), self.reg()
        if kind == "r":
            return f"{self.rng.choice(list(R_OPS))} x{rd}, x{rs1}, x{rs2}"
        if kind == "i":
            return f"{self.rng.choice(list(I_OPS))} x{rd}, x{rs1}, {self.imm()}"
        if kind == "shift":
            return f"{self.rng.choice(list(SHIFT_OPS))} x{rd}, x{rs1}, {self.rng.randrange(32)}"
        return f"{kind} x{rd}, {self.rng.randrange(1 << 20)}"

    def memory(self) -> str:
        if self.rng.random() < 0.5:
            op = self.rng.choice(list(LOAD_OPS))
            size = 1 << (LOAD_OPS[op] & 3)
            return f"{op} x{self.reg()}, {self.rng.randrange(0, 2048, size)}(x31)"
        op = self.rng.choice(list(STORE_OPS))
        size = 1 << STORE_OPS[op]
        return f"{op} x{self.reg()}, {self.rng.randrange(0, 2048, size)}(x31)"

    def straight(self) -> str:
        return self.memory() if self.rng.random() < 0.3 else self.alu()

    def build(self, items: int, weights: dict[str, int]) -> None:
        self.lines.append(f"li x31, {DATA}")
        for i in range(32):
            self.lines += [f"li x1, {self.value()}", f"sw x1, {4 * i}(x31)"]
        self.lines += [f"li x{n}, {self.value()}" for n in range(1, 16)]
        kinds, counts = zip(*weights.items())
        for _ in range(items):
            getattr(self, self.rng.choices(kinds, counts)[0])()

    def emit_alu(self) -> None:
        self.lines.append(self.alu())

    def emit_memory(self) -> None:
        self.lines.append(self.memory())

    def emit_branch(self) -> None:
        end = self.label()
        op = self.rng.choice(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
        a = self.reg()
        b = a if self.rng.random() < 0.2 else self.reg()
        self.lines.append(f"{op} x{a}, x{b}, {end}")
        self.lines += [self.straight() for _ in range(self.rng.randint(1, 3))]
        self.lines.append(f"{end}:")

    def emit_jal(self) -> None:
        end = self.label()
        self.lines.append(f"jal x{self.reg()}, {end}")
        self.lines += [self.straight() for _ in range(self.rng.randint(1, 3))]
        self.lines.append(f"{end}:")

    def emit_jalr(self) -> None:
        skipped = self.rng.randint(1, 3)
        offset = self.rng.choice([0, 1, 8, -8])
        rt = self.rng.randrange(1, 16)
        self.lines += [
            f"auipc x{rt}, 0",
            f"addi x{rt}, x{rt}, {12 + 4 * skipped - offset}",
            f"jalr x{self.reg()}, {offset}(x{rt})",
        ]
        self.lines += [self.straight() for _ in range(skipped)]

    def emit_loop(self) -> None:
        top = self.label()
        self.lines += [f"li x16, {self.rng.randint(2, 6)}", f"{top}:"]
        self.lines += [self.straight() for _ in range(self.rng.randint(1, 3))]
        self.lines += ["addi x16, x16, -1", f"bnez x16, {top}"]

    def source(self) -> str:
        return "\n".join(self.lines + ["ecall"])


MIXES = {
    "mixed": dict(emit_alu=30, emit_memory=14, emit_branch=8, emit_jal=3, emit_jalr=3, emit_loop=3),
    "alu": dict(emit_alu=1),
    "memory": dict(emit_alu=12, emit_memory=30, emit_loop=2),
    "control": dict(emit_alu=10, emit_branch=14, emit_jal=6, emit_jalr=6, emit_loop=5),
}


def random_programs() -> list[Program]:
    plan = [("mixed", seed, 600) for seed in range(1, 6)] + [("alu", 11, 900), ("memory", 12, 700), ("control", 13, 500)]
    return [
        Program(f"random_{mix}_{seed}", RandomProgram(seed, items, MIXES[mix]).source(), {31: DATA, 0: 0})
        for mix, seed, items in plan
    ]


def all_programs() -> list[Program]:
    return [
        alu_basic(), upper_immediates(), slt_edges(), shifts(), branches(), branch_far(), jumps(), loads(), stores(),
        x0_behaviour(), hazards(), *halt_stops(), fence_nop(), memory_edges(), *loop_programs(), recursion(),
        *random_programs(),
    ]


def main() -> int:
    check_assembler()
    OUT_DIR.mkdir(exist_ok=True)
    for stale in [*OUT_DIR.glob("*.hex"), *OUT_DIR.glob("*.expected")]:
        stale.unlink()
    programs = all_programs()
    for program in programs:
        write_program(program)
    print(f"[gen_tests::info] wrote {len(programs)} programs to {OUT_DIR}, size limit {MEM_BYTES} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
