// Single-cycle RV32I: one instruction per clock, combinational memory read,
// registers, memory and pc updated on the rising edge.
module rv32i_top (
    input  logic clk,
    input  logic rst,
    output logic halted
);
    logic [31:0] mem  [0:16383];
    logic [31:0] regs [0:31];
    logic [31:0] pc;

    localparam logic [6:0] OP_LUI    = 7'b0110111;
    localparam logic [6:0] OP_AUIPC  = 7'b0010111;
    localparam logic [6:0] OP_JAL    = 7'b1101111;
    localparam logic [6:0] OP_JALR   = 7'b1100111;
    localparam logic [6:0] OP_BRANCH = 7'b1100011;
    localparam logic [6:0] OP_LOAD   = 7'b0000011;
    localparam logic [6:0] OP_STORE  = 7'b0100011;
    localparam logic [6:0] OP_IMM    = 7'b0010011;
    localparam logic [6:0] OP_REG    = 7'b0110011;

    // Fetch and decode
    wire [31:0] instr  = mem[pc[15:2]];
    wire [6:0]  opcode = instr[6:0];
    wire [4:0]  rd     = instr[11:7];
    wire [2:0]  funct3 = instr[14:12];
    wire [4:0]  rs1    = instr[19:15];
    wire [4:0]  rs2    = instr[24:20];
    wire        alt    = instr[30];

    wire [31:0] imm_i = {{20{instr[31]}}, instr[31:20]};
    wire [31:0] imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};
    wire [31:0] imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    wire [31:0] imm_u = {instr[31:12], 12'b0};
    wire [31:0] imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

    wire [31:0] rs1_val = (rs1 == 5'd0) ? 32'd0 : regs[rs1];
    wire [31:0] rs2_val = (rs2 == 5'd0) ? 32'd0 : regs[rs2];

    wire is_ecall = (instr == 32'h00000073);

    // ALU: OP-IMM and OP share funct3; only OP can subtract
    wire [31:0] alu_b = (opcode == OP_REG) ? rs2_val : imm_i;
    wire [4:0]  shamt = alu_b[4:0];
    logic [31:0] alu_out;
    always_comb begin
        case (funct3)
            3'b000:  alu_out = (opcode == OP_REG && alt) ? rs1_val - alu_b : rs1_val + alu_b;
            3'b001:  alu_out = rs1_val << shamt;
            3'b010:  alu_out = {31'b0, $signed(rs1_val) < $signed(alu_b)};
            3'b011:  alu_out = {31'b0, rs1_val < alu_b};
            3'b100:  alu_out = rs1_val ^ alu_b;
            3'b101:  alu_out = alt ? 32'($signed(rs1_val) >>> shamt) : rs1_val >> shamt;
            3'b110:  alu_out = rs1_val | alu_b;
            default: alu_out = rs1_val & alu_b;
        endcase
    end

    // Branch condition
    logic taken;
    always_comb begin
        case (funct3)
            3'b000:  taken = rs1_val == rs2_val;
            3'b001:  taken = rs1_val != rs2_val;
            3'b100:  taken = $signed(rs1_val) < $signed(rs2_val);
            3'b101:  taken = $signed(rs1_val) >= $signed(rs2_val);
            3'b110:  taken = rs1_val < rs2_val;
            3'b111:  taken = rs1_val >= rs2_val;
            default: taken = 1'b0;
        endcase
    end

    // Data memory: address is rs1 + imm, accesses are naturally aligned
    wire [31:0] addr      = rs1_val + (opcode == OP_STORE ? imm_s : imm_i);
    wire [31:0] mem_word  = mem[addr[15:2]];
    wire [4:0]  bit_shift = {addr[1:0], 3'b000};
    wire [31:0] shifted   = mem_word >> bit_shift;

    logic [31:0] load_val;
    always_comb begin
        case (funct3)
            3'b000:  load_val = {{24{shifted[7]}}, shifted[7:0]};
            3'b001:  load_val = {{16{shifted[15]}}, shifted[15:0]};
            3'b100:  load_val = {24'b0, shifted[7:0]};
            3'b101:  load_val = {16'b0, shifted[15:0]};
            default: load_val = mem_word;
        endcase
    end

    logic [31:0] store_mask;
    always_comb begin
        case (funct3[1:0])
            2'b00:   store_mask = 32'h000000FF << bit_shift;
            2'b01:   store_mask = 32'h0000FFFF << bit_shift;
            default: store_mask = 32'hFFFFFFFF;
        endcase
    end
    wire [31:0] store_word = (mem_word & ~store_mask) | ((rs2_val << bit_shift) & store_mask);

    // Writeback value and next pc
    wire [31:0] pc_plus4 = pc + 32'd4;
    logic [31:0] rd_val;
    logic        rd_we;
    logic [31:0] next_pc;
    always_comb begin
        rd_val  = alu_out;
        rd_we   = 1'b0;
        next_pc = pc_plus4;
        case (opcode)
            OP_LUI:    begin rd_val = imm_u;      rd_we = 1'b1; end
            OP_AUIPC:  begin rd_val = pc + imm_u; rd_we = 1'b1; end
            OP_JAL:    begin rd_val = pc_plus4;   rd_we = 1'b1; next_pc = pc + imm_j; end
            OP_JALR:   begin rd_val = pc_plus4;   rd_we = 1'b1; next_pc = (rs1_val + imm_i) & ~32'd1; end
            OP_BRANCH: if (taken) next_pc = pc + imm_b;
            OP_LOAD:   begin rd_val = load_val;   rd_we = 1'b1; end
            OP_IMM,
            OP_REG:    rd_we = 1'b1;
            default:   ;
        endcase
    end

    wire store_en = (opcode == OP_STORE);
    wire halt_now = is_ecall;

    always_ff @(posedge clk) begin
        if (rst) begin
            pc     <= 32'd0;
            halted <= 1'b0;
            for (int i = 0; i < 32; i++) regs[i] <= 32'd0;
        end else if (!halted) begin
            if (halt_now) begin
                halted <= 1'b1;
            end else begin
                pc <= next_pc;
                if (rd_we && rd != 5'd0) regs[rd] <= rd_val;
                if (store_en) mem[addr[15:2]] <= store_word;
            end
        end
    end
endmodule
