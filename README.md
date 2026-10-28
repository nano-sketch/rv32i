# rv32i

Single-cycle 32-bit RISC-V CPU (RV32I) in SystemVerilog.

- [x] All of RV32I: LUI, AUIPC, JAL, JALR, branches, loads, stores, ALU ops, FENCE (no-op), ECALL
- [x] Single-cycle: one instruction per clock, combinational memory read, writes on the rising edge
- [x] 64 KiB unified memory (`mem`) and register file (`regs`) inside `rv32i_top`
- [x] ECALL sets `halted` and freezes pc, registers and memory
- [ ] Misaligned accesses, traps, CSRs, interrupts (not supported)

## Design

`rtl/rv32i_top.sv` is the whole core. `rst` is synchronous and active high: it clears `pc`, `regs` and `halted`, and leaves `mem` alone so a testbench can load a program first. Execution starts at address 0. Stores read-modify-write the containing word.

## Run the tests

```
cd rv32i
python run_tests.py
```

Build by hand:

```
C:\iverilog\bin\iverilog.exe -g2012 -s rv32i_top -o sim.vvp rtl/*.sv
```

## Not supported

- Misaligned loads and stores (programs must align them)
- Addresses at or above `0x10000`
- Traps, exceptions, CSRs, interrupts; any `ECALL` simply halts
- `EBREAK` and other SYSTEM instructions are treated as no-ops

## License

MIT
