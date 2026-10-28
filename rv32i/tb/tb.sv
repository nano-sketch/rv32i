// Runs one program: +hex=<file> loads dut.mem, +out=<file> receives the final state.
// Dump format: "x<n> <hex8>" for the 32 registers, then "m<word index hex4> <hex8>" for all 16384 memory words.
module tb;
    localparam int MAX_CYCLES = 1_000_000;
    localparam int EXTRA_CYCLES = 8;

    logic clk = 0;
    logic rst = 1;
    logic halted;

    rv32i_top dut (.clk(clk), .rst(rst), .halted(halted));

    always #5 clk = ~clk;

    logic [31:0] snap_regs [0:31];
    logic [31:0] snap_mem [0:16383];

    reg [8*256-1:0] hex_path;
    reg [8*256-1:0] out_path;
    integer cycles;
    integer fd;
    integer i;
    bit changed;

    initial begin
        if (!$value$plusargs("hex=%s", hex_path) || !$value$plusargs("out=%s", out_path)) begin
            $display("[tb::error] usage: vvp sim.vvp +hex=<program.hex> +out=<dump.txt>");
            $finish;
        end

        for (i = 0; i < 16384; i = i + 1) dut.mem[i] = 32'h0;
        $readmemh(hex_path, dut.mem);

        repeat (4) @(posedge clk);
        #1 rst = 0;

        cycles = 0;
        while (!halted && cycles < MAX_CYCLES) begin
            @(posedge clk);
            #1 cycles = cycles + 1;
        end
        if (!halted) begin
            $display("[tb::error] timeout: not halted after %0d cycles", MAX_CYCLES);
            $finish;
        end

        for (i = 0; i < 32; i = i + 1) snap_regs[i] = dut.regs[i];
        for (i = 0; i < 16384; i = i + 1) snap_mem[i] = dut.mem[i];
        repeat (EXTRA_CYCLES) @(posedge clk);
        #1;

        changed = 0;
        for (i = 0; i < 32; i = i + 1) changed |= (dut.regs[i] !== snap_regs[i]);
        for (i = 0; i < 16384; i = i + 1) changed |= (dut.mem[i] !== snap_mem[i]);
        if (changed || !halted) $display("[tb::error] state changed after halted");

        fd = $fopen(out_path, "w");
        if (fd == 0) begin
            $display("[tb::error] cannot open %0s", out_path);
            $finish;
        end
        for (i = 0; i < 32; i = i + 1) $fdisplay(fd, "x%0d %08x", i, dut.regs[i]);
        for (i = 0; i < 16384; i = i + 1) $fdisplay(fd, "m%04x %08x", i, dut.mem[i]);
        $fclose(fd);
        $display("[tb::info] halted after %0d cycles", cycles);
        $finish;
    end
endmodule
