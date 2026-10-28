"""RV32I assembler: assemble(text) -> list of 32-bit words, starting at address 0.

Syntax: one instruction per line, "label:" prefixes, '#' or ';' comments, numbers in decimal or 0x hex.
Pseudo-instructions: nop, li, mv, not, neg, j, jr, ret, call, beqz, bnez, .word.
Branch and jump operands are labels or numeric byte offsets.
"""
import re

ABI_NAMES = (
    "zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 a6 a7 "
    "s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 t3 t4 t5 t6"
).split()
REGISTERS = {f"x{n}": n for n in range(32)} | {name: n for n, name in enumerate(ABI_NAMES)} | {"fp": 8}

R_OPS = {  # name: (funct7, funct3)
    "add": (0x00, 0), "sub": (0x20, 0), "sll": (0x00, 1), "slt": (0x00, 2), "sltu": (0x00, 3),
    "xor": (0x00, 4), "srl": (0x00, 5), "sra": (0x20, 5), "or": (0x00, 6), "and": (0x00, 7),
}
I_OPS = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
SHIFT_OPS = {"slli": (0x00, 1), "srli": (0x00, 5), "srai": (0x20, 5)}
LOAD_OPS = {"lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
STORE_OPS = {"sb": 0, "sh": 1, "sw": 2}
BRANCH_OPS = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}

OP_LUI, OP_AUIPC, OP_JAL, OP_JALR = 0x37, 0x17, 0x6F, 0x67
OP_BRANCH, OP_LOAD, OP_STORE, OP_IMM, OP_REG = 0x63, 0x03, 0x23, 0x13, 0x33
ECALL, FENCE = 0x00000073, 0x0FF0000F


class AsmError(Exception):
    pass


def _fits(value: int, bits: int) -> bool:
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


def _check_imm(value: int, bits: int, what: str) -> int:
    if not _fits(value, bits):
        raise AsmError(f"[asm::error] {what} {value} does not fit in {bits} signed bits")
    return value & ((1 << bits) - 1)


def encode_r(name: str, rd: int, rs1: int, rs2: int) -> int:
    funct7, funct3 = R_OPS[name]
    return funct7 << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | OP_REG


def encode_i(opcode: int, funct3: int, rd: int, rs1: int, imm: int) -> int:
    return _check_imm(imm, 12, "immediate") << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode


def encode_shift(name: str, rd: int, rs1: int, shamt: int) -> int:
    if not 0 <= shamt < 32:
        raise AsmError(f"[asm::error] shift amount {shamt} out of range")
    funct7, funct3 = SHIFT_OPS[name]
    return funct7 << 25 | shamt << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | OP_IMM


def encode_s(name: str, rs2: int, rs1: int, imm: int) -> int:
    imm = _check_imm(imm, 12, "offset")
    return (imm >> 5) << 25 | rs2 << 20 | rs1 << 15 | STORE_OPS[name] << 12 | (imm & 0x1F) << 7 | OP_STORE


def encode_b(name: str, rs1: int, rs2: int, offset: int) -> int:
    if offset & 1:
        raise AsmError(f"[asm::error] branch offset {offset} is odd")
    imm = _check_imm(offset, 13, "branch offset")
    return (
        (imm >> 12) << 31 | ((imm >> 5) & 0x3F) << 25 | rs2 << 20 | rs1 << 15
        | BRANCH_OPS[name] << 12 | ((imm >> 1) & 0xF) << 8 | ((imm >> 11) & 1) << 7 | OP_BRANCH
    )


def encode_u(opcode: int, rd: int, imm20: int) -> int:
    return (imm20 & 0xFFFFF) << 12 | rd << 7 | opcode


def encode_j(rd: int, offset: int) -> int:
    if offset & 1:
        raise AsmError(f"[asm::error] jump offset {offset} is odd")
    imm = _check_imm(offset, 21, "jump offset")
    return (
        (imm >> 20) << 31 | ((imm >> 1) & 0x3FF) << 21 | ((imm >> 11) & 1) << 20
        | ((imm >> 12) & 0xFF) << 12 | rd << 7 | OP_JAL
    )


def _reg(token: str) -> int:
    try:
        return REGISTERS[token]
    except KeyError:
        raise AsmError(f"[asm::error] unknown register '{token}'") from None


def _num(token: str) -> int:
    try:
        return int(token, 0)
    except ValueError:
        raise AsmError(f"[asm::error] bad number '{token}'") from None


def _mem_operand(token: str) -> tuple[int, int]:
    match = re.fullmatch(r"(-?\w*)\((\w+)\)", token)
    if not match:
        raise AsmError(f"[asm::error] bad memory operand '{token}'")
    return _num(match.group(1) or "0"), _reg(match.group(2))


def split_li(value: int) -> tuple[int, int]:
    """Upper 20 bits and signed low 12 bits such that (upper << 12) + low == value mod 2^32."""
    value &= 0xFFFFFFFF
    low = value & 0xFFF
    if low >= 0x800:
        low -= 0x1000
    return ((value - low) >> 12) & 0xFFFFF, low


def _expand(mnemonic: str, ops: list[str]) -> list[tuple[str, list[str]]]:
    """Rewrite pseudo-instructions into real ones; the size of each is fixed so labels resolve in one pass."""
    if mnemonic == "nop":
        return [("addi", ["x0", "x0", "0"])]
    if mnemonic == "mv":
        return [("addi", [ops[0], ops[1], "0"])]
    if mnemonic == "not":
        return [("xori", [ops[0], ops[1], "-1"])]
    if mnemonic == "neg":
        return [("sub", [ops[0], "x0", ops[1]])]
    if mnemonic == "j":
        return [("jal", ["x0", ops[0]])]
    if mnemonic == "jr":
        return [("jalr", ["x0", f"0({ops[0]})"])]
    if mnemonic == "ret":
        return [("jalr", ["x0", "0(ra)"])]
    if mnemonic == "call":
        return [("jal", ["ra", ops[0]])]
    if mnemonic == "beqz":
        return [("beq", [ops[0], "x0", ops[1]])]
    if mnemonic == "bnez":
        return [("bne", [ops[0], "x0", ops[1]])]
    if mnemonic == "li":
        value = _num(ops[1])
        if _fits(value, 12):
            return [("addi", [ops[0], "x0", str(value)])]
        upper, low = split_li(value)
        return [("lui", [ops[0], str(upper)]), ("addi", [ops[0], ops[0], str(low)])]
    return [(mnemonic, ops)]


def _encode(mnemonic: str, ops: list[str], pc: int, labels: dict[str, int]) -> int:
    def target(token: str) -> int:
        return labels[token] - pc if token in labels else _num(token)

    if mnemonic == ".word":
        return _num(ops[0]) & 0xFFFFFFFF
    if mnemonic == "ecall":
        return ECALL
    if mnemonic == "fence":
        return FENCE
    if mnemonic in R_OPS:
        return encode_r(mnemonic, _reg(ops[0]), _reg(ops[1]), _reg(ops[2]))
    if mnemonic in I_OPS:
        return encode_i(OP_IMM, I_OPS[mnemonic], _reg(ops[0]), _reg(ops[1]), _num(ops[2]))
    if mnemonic in SHIFT_OPS:
        return encode_shift(mnemonic, _reg(ops[0]), _reg(ops[1]), _num(ops[2]))
    if mnemonic in LOAD_OPS:
        offset, base = _mem_operand(ops[1])
        return encode_i(OP_LOAD, LOAD_OPS[mnemonic], _reg(ops[0]), base, offset)
    if mnemonic in STORE_OPS:
        offset, base = _mem_operand(ops[1])
        return encode_s(mnemonic, _reg(ops[0]), base, offset)
    if mnemonic in BRANCH_OPS:
        return encode_b(mnemonic, _reg(ops[0]), _reg(ops[1]), target(ops[2]))
    if mnemonic == "lui":
        return encode_u(OP_LUI, _reg(ops[0]), _num(ops[1]))
    if mnemonic == "auipc":
        return encode_u(OP_AUIPC, _reg(ops[0]), _num(ops[1]))
    if mnemonic == "jal":
        return encode_j(_reg(ops[0]), target(ops[1]))
    if mnemonic == "jalr":
        offset, base = _mem_operand(ops[1])
        return encode_i(OP_JALR, 0, _reg(ops[0]), base, offset)
    raise AsmError(f"[asm::error] unknown instruction '{mnemonic}'")


def assemble(text: str) -> list[int]:
    labels: dict[str, int] = {}
    items: list[tuple[int, int, str, list[str]]] = []  # (line number, pc, mnemonic, operands)
    pc = 0
    for number, raw in enumerate(text.splitlines(), 1):
        line = re.split(r"[#;]", raw)[0].strip()
        while match := re.match(r"(\w+):\s*", line):
            if match.group(1) in labels:
                raise AsmError(f"[asm::error] line {number}: label '{match.group(1)}' defined twice")
            labels[match.group(1)] = pc
            line = line[match.end():]
        if not line:
            continue
        mnemonic, _, rest = line.partition(" ")
        operands = [op.strip() for op in rest.split(",") if op.strip()]
        for real_mnemonic, real_ops in _expand(mnemonic.lower(), operands):
            items.append((number, pc, real_mnemonic, real_ops))
            pc += 4

    words = []
    for number, pc, mnemonic, ops in items:
        try:
            words.append(_encode(mnemonic, ops, pc, labels))
        except (AsmError, IndexError, KeyError) as error:
            raise AsmError(f"line {number}: {mnemonic} {', '.join(ops)}: {error!r}") from None
    return words
