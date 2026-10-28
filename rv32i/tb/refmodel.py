"""Reference RV32I simulator that decodes machine code words; used to compute the expected final state."""

MEM_WORDS = 16384
MEM_BYTES = MEM_WORDS * 4
MASK = 0xFFFFFFFF


class RefError(Exception):
    pass


def sext(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


class Machine:
    def __init__(self, program: list[int]) -> None:
        if len(program) > MEM_WORDS:
            raise RefError("[refmodel::error] program larger than memory")
        self.mem = list(program) + [0] * (MEM_WORDS - len(program))
        self.regs = [0] * 32
        self.pc = 0
        self.halted = False

    def _check(self, addr: int, size: int) -> None:
        if addr % size or not 0 <= addr <= MEM_BYTES - size:
            raise RefError(f"[refmodel::error] bad {size}-byte access at {addr:#X} (pc {self.pc:#X})")

    def load(self, addr: int, size: int) -> int:
        self._check(addr, size)
        shift = (addr & 3) * 8
        return (self.mem[addr >> 2] >> shift) & ((1 << (8 * size)) - 1)

    def store(self, addr: int, size: int, value: int) -> None:
        self._check(addr, size)
        shift = (addr & 3) * 8
        mask = ((1 << (8 * size)) - 1) << shift
        word = self.mem[addr >> 2]
        self.mem[addr >> 2] = (word & ~mask & MASK) | ((value << shift) & mask)

    def write(self, rd: int, value: int) -> None:
        if rd:
            self.regs[rd] = value & MASK

    def step(self) -> None:
        pc = self.pc
        inst = self.load(pc, 4)
        opcode, rd = inst & 0x7F, inst >> 7 & 0x1F
        funct3, rs1, rs2, funct7 = inst >> 12 & 7, inst >> 15 & 0x1F, inst >> 20 & 0x1F, inst >> 25
        a, b = self.regs[rs1], self.regs[rs2]
        sa, sb = sext(a, 32), sext(b, 32)
        imm_i = sext(inst >> 20, 12)
        next_pc = pc + 4

        if opcode == 0x37:  # LUI
            self.write(rd, inst & 0xFFFFF000)
        elif opcode == 0x17:  # AUIPC
            self.write(rd, pc + (inst & 0xFFFFF000))
        elif opcode == 0x6F:  # JAL
            imm = sext((inst >> 31) << 20 | (inst >> 12 & 0xFF) << 12 | (inst >> 20 & 1) << 11 | (inst >> 21 & 0x3FF) << 1, 21)
            self.write(rd, pc + 4)
            next_pc = pc + imm
        elif opcode == 0x67 and funct3 == 0:  # JALR
            next_pc = (a + imm_i) & 0xFFFFFFFE
            self.write(rd, pc + 4)
        elif opcode == 0x63:  # branches
            imm = sext((inst >> 31) << 12 | (inst >> 7 & 1) << 11 | (inst >> 25 & 0x3F) << 5 | (inst >> 8 & 0xF) << 1, 13)
            taken = {0: a == b, 1: a != b, 4: sa < sb, 5: sa >= sb, 6: a < b, 7: a >= b}.get(funct3)
            if taken is None:
                raise RefError(f"[refmodel::error] bad branch funct3 at {pc:#X}")
            if taken:
                next_pc = pc + imm
        elif opcode == 0x03:  # loads
            addr = (a + imm_i) & MASK
            size = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}.get(funct3)
            if size is None:
                raise RefError(f"[refmodel::error] bad load funct3 at {pc:#X}")
            value = self.load(addr, size)
            self.write(rd, sext(value, 8 * size) if funct3 < 4 else value)
        elif opcode == 0x23:  # stores
            addr = (a + sext(funct7 << 5 | rd, 12)) & MASK
            size = {0: 1, 1: 2, 2: 4}.get(funct3)
            if size is None:
                raise RefError(f"[refmodel::error] bad store funct3 at {pc:#X}")
            self.store(addr, size, b)
        elif opcode == 0x13:  # register-immediate
            shamt = rs2
            if funct3 == 0:
                result = a + imm_i
            elif funct3 == 2:
                result = int(sa < imm_i)
            elif funct3 == 3:
                result = int(a < imm_i & MASK)
            elif funct3 == 4:
                result = a ^ imm_i
            elif funct3 == 6:
                result = a | imm_i
            elif funct3 == 7:
                result = a & imm_i
            elif funct3 == 1 and funct7 == 0:
                result = a << shamt
            elif funct3 == 5 and funct7 == 0:
                result = a >> shamt
            elif funct3 == 5 and funct7 == 0x20:
                result = sa >> shamt
            else:
                raise RefError(f"[refmodel::error] bad op-imm at {pc:#X}")
            self.write(rd, result)
        elif opcode == 0x33:  # register-register
            shamt = b & 31
            ops = {
                (0, 0): a + b, (0x20, 0): a - b, (0, 1): a << shamt, (0, 2): int(sa < sb), (0, 3): int(a < b),
                (0, 4): a ^ b, (0, 5): a >> shamt, (0x20, 5): sa >> shamt, (0, 6): a | b, (0, 7): a & b,
            }
            if (funct7, funct3) not in ops:
                raise RefError(f"[refmodel::error] bad op at {pc:#X}")
            self.write(rd, ops[funct7, funct3])
        elif opcode == 0x0F:  # FENCE
            pass
        elif inst == 0x00000073:  # ECALL
            self.halted = True
            return
        else:
            raise RefError(f"[refmodel::error] illegal instruction {inst:08X} at {pc:#X}")
        self.pc = next_pc & MASK


def run(program: list[int], max_steps: int = 1_000_000) -> Machine:
    """Execute until ECALL; returns the machine with the final registers and memory."""
    machine = Machine(program)
    for _ in range(max_steps):
        machine.step()
        if machine.halted:
            return machine
    raise RefError(f"[refmodel::error] no ECALL within {max_steps} steps")
