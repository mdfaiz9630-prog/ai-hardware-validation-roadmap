# ai-hardware-validation-roadmap
12-month roadmap for SoC &amp; AI hardware validation
# AI Hardware Validation: Complete 12-Month Roadmap for Freshers

**Author:** AI Learning Path  
**Version:** 1.0 - January 2026  
**Target Audience:** Final-year ECE Students / Freshers  
**Duration:** 12 months | 10-15 hours/week  

---

## TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Why This Roadmap](#why-this-roadmap)
3. [Month-by-Month Overview](#month-by-month-overview)
4. [Detailed Week-by-Week Schedule](#detailed-week-by-week-schedule)
5. [Mini-Projects](#mini-projects)
6. [Resources & Links](#resources--links)
7. [Success Metrics](#success-metrics)
8. [Getting Started Checklist](#getting-started-checklist)

---

## EXECUTIVE SUMMARY

This is a **comprehensive, self-paced learning path** designed for freshers to build expertise in:
- **Digital Electronics** (foundation)
- **Computer Architecture** (building on it)
- **Hardware Verification** (testing designs)
- **AI/ML Fundamentals** (understanding workloads)
- **AI Hardware Validation** (ultimate goal)

### By the end of 12 months, you will:
✅ Design digital circuits from logic gates to CPUs  
✅ Write comprehensive hardware verification testbenches  
✅ Understand neural networks and AI workloads  
✅ Design and validate a custom AI accelerator  
✅ Have a portfolio of 5+ hardware projects on GitHub  

---

## WHY THIS ROADMAP

### The Problem
Most students learn these subjects separately. You'll find:
- Digital Electronics courses without validation focus
- Architecture courses without AI context
- ML courses without hardware understanding

**This roadmap connects them all.**

### The Solution
**Validation-Driven Learning:** Each phase culminates in testing & debugging real circuits against specifications.

---

## MONTH-BY-MONTH OVERVIEW

```
MONTH   1-2          3-4              5-6            7-8        9-10         11-12
        ╔═══╗        ╔═══╗            ╔═══╗          ╔═══╗      ╔════╗        ╔════╗
        ║DIG║        ║ARC║            ║VER║          ║ AI║      ║CONN║        ║VALID║
        ║ELC║──────→ ║HIT║──────────→ ║IF ║────────→║/ML║────→║ECT ║───────→║ATION║
        ║   ║        ║ECT║            ║   ║          ║   ║      ║    ║        ║     ║
        ║(P1)║       ║(P4)║           ║(P5)║         ║(P6)║      ║(P7,8)║      ║(P9) ║
        ╚═══╝        ╚═══╝            ╚═══╝          ╚═══╝      ╚════╝        ╚════╝
        
P1-P9 = Mini-Projects with working deliverables
```

| Phase | Months | Topic | Mini-Project | Time/Week |
|-------|--------|-------|--------------|-----------|
| 1 | 1-2 | Digital Electronics | P1: 4-bit Adder | 10-12 hrs |
| 2 | 3-4 | Computer Architecture | P4: 5-Stage Pipeline CPU | 10-12 hrs |
| 3 | 5-6 | Hardware Verification | P5: Comprehensive Testbench | 10-12 hrs |
| 4 | 7-8 | AI/ML Fundamentals | P6: Train Neural Network | 8-10 hrs |
| 5 | 9-10 | Hardware for AI | P7,P8: Systolic Array + Analysis | 10-12 hrs |
| 6 | 11-12 | AI Hardware Validation | P9: Complete AI Accelerator | 12-15 hrs |

---

## DETAILED WEEK-BY-WEEK SCHEDULE

### MONTHS 1-2: DIGITAL ELECTRONICS FUNDAMENTALS

**Goal:** Understand digital logic, circuits, and state machines

#### **Week 1-2: Number Systems & Logic Gates (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 1 | Binary, Boolean Algebra, Logic Gates | 2-3 hrs | Khan Academy, CircuitBread | Truth tables for 8 gates |
| 2 | Gate Combinations, Simplification | 2-3 hrs | All About Circuits, Nandland | Simplify 3 Boolean expressions |

**Resources:**
- 🎥 **Khan Academy**: "Binary System" + "Boolean Algebra" (3 hours)
  - Link: https://www.khanacademy.org/
- 📚 **CircuitBread**: "Logic Gates" tutorial (2 hours)
  - Link: https://www.circuitbread.com/tutorials/logic-gates
- 📖 **All About Circuits**: Chapter 2-3 (2 hours)
  - Link: https://www.allaboutcircuits.com/textbook/digital/
- 🎥 **Nandland** (YouTube): "Digital Logic" series (2 hours)
  - Search: "Nandland digital logic"

**Mini-Assessment:**
- [ ] Convert decimal 255 to binary and back
- [ ] Create truth table for XOR gate
- [ ] Simplify: (A + B) · (¬A + C)

---

#### **Week 3-4: Combinational Circuits (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 3 | Multiplexers, Decoders, Adders | 2-3 hrs | CircuitBread, YouTube | Design 4-bit adder |
| 4 | Subtractors, Comparators, ALU | 2-3 hrs | All About Circuits | Design simple 1-bit ALU |

**Mini-Project 1: 4-Bit Binary Adder**
- Understand: 1-bit full adder → 4-bit cascaded
- Deliverable: Schematic + 5 test cases
- Time: 6-8 hours total

---

#### **Week 5-6: Sequential Logic (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 5 | Flip-Flops, Registers, Counters | 2-3 hrs | CircuitBread, All About Circuits | Design 8-bit counter |
| 6 | Finite State Machines, Memory | 2-3 hrs | YouTube, Textbooks | Draw FSM for traffic light |

**Mini-Project 2: 8-Bit Up/Down Counter**
- Design with Enable and Direction inputs
- Deliverable: Logic diagram + timing diagrams
- Time: 8-10 hours total

---

#### **Week 7-8: Advanced Sequential & Review (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 7 | Advanced FSM, Registers, Memory | 2-3 hrs | Textbooks + Practice | Complex state machine |
| 8 | Review, Assessment, Planning | 2-3 hrs | Self-quizzes | Month 1-2 Summary |

**Phase 1 Assessment:**
- [ ] Design 8-bit adder with overflow detection
- [ ] Design modulo-12 counter
- [ ] Explain difference between latch and flip-flop
- [ ] Create FSM for simple controller

---

### MONTHS 3-4: COMPUTER ARCHITECTURE

**Goal:** Understand CPU design, pipelines, and memory systems

#### **Week 9-10: Processor Fundamentals (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 9 | Instruction Sets, RISC-V Assembly | 2-3 hrs | MIT OCW, RISC-V spec | Write 20-line RISC-V program |
| 10 | Datapath, Control Unit, ALU | 2-3 hrs | MIT OCW 6.823, NPTEL | Design simple datapath |

**Resources:**
- 🎓 **MIT OpenCourseWare 6.823**: Computer System Architecture
  - Link: https://ocw.mit.edu/courses/6-823-computer-system-architecture-fall-2005/
  - Watch: Lectures 1-6 (6 hours)
- 🎓 **NPTEL IIT Delhi**: Computer Architecture by Prof. Sarangi
  - Link: https://nptel.ac.in/
  - Watch: Lectures 1-10 (10 hours)
- 📖 **RISC-V Specification**: Free online
  - Link: https://riscv.org/
- 📘 **Patterson & Hennessy**: "Computer Organization and Design" (RISC-V Edition)
  - Check library or free online resources

**Mini-Project 3: RISC-V Program**
- Implement: sum, difference, product of two numbers
- In: Assembly language (50+ lines)
- Test: With simple test cases
- Time: 6-8 hours

---

#### **Week 11-12: Pipelining (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 11 | 5-Stage Pipeline, Instruction Flow | 2-3 hrs | MIT OCW 6.823, YouTube | Trace 4 instructions through pipeline |
| 12 | Pipeline Hazards, Performance Metrics | 2-3 hrs | MIT OCW, NPTEL | Identify and resolve hazards |

**Key Concepts:**
```
Pipeline Stages: Fetch → Decode → Execute → Memory → Write-back
Hazards: Data (RAW, WAW, WAR), Control (branches)
Solutions: Forwarding, stalling, branch prediction
Performance: CPI = 1 + Stalls/Instruction
```

---

#### **Week 13-14: Memory Hierarchy (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 13 | Cache, Memory Hierarchy, Bandwidth | 2-3 hrs | MIT OCW, NPTEL | Calculate cache behavior |
| 14 | Virtual Memory, Performance Modeling | 2-3 hrs | MIT OCW, Roofline Model | Create roofline diagram |

**Mini-Project 4: 5-Stage Pipeline CPU (Verilog)**
- Design: Fetch, Decode, Execute, Memory, Write-back stages
- Implement: Control unit, datapath, hazard handling
- Test: 15-instruction RISC-V program
- Deliverable: ~400 lines Verilog + testbench
- Time: 20-30 hours (across weeks 9-14)

---

### MONTHS 5-6: HARDWARE VERIFICATION

**Goal:** Learn to write comprehensive, professional testbenches

#### **Week 15-16: Verilog HDL (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 15 | Verilog Syntax, Modules, Data Types | 2-3 hrs | ChipVerify, EDA Playground | Write 5 Verilog modules |
| 16 | Always Blocks, Parameters, Instances | 2-3 hrs | ChipVerify, HDLBits | Code parameterized n-bit adder |

**Free Tools:**
- 🌐 **EDA Playground**: https://edaplayground.com/
  - No installation, instant simulation
  - Verilog & SystemVerilog support
  - Share designs easily
- 🎯 **HDLBits**: https://hdlbits.com/
  - Interactive Verilog problems
  - Progressive difficulty
  - Instant feedback

**Resources:**
- 📖 **ChipVerify Verilog Tutorial**: Beginner to advanced
  - Link: https://www.chipverify.com/verilog/verilog-tutorials
- 📖 **All About Circuits**: Digital design chapters
  - Link: https://www.allaboutcircuits.com/textbook/digital/
- 📘 **"Verilog HDL: A Guide to Digital Design and Synthesis"** by Samir Palnitkar
  - Check library or free online

---

#### **Week 17-18: Testbenches & Stimulus (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 17 | Testbench Structure, Stimulus Generation | 2-3 hrs | ChipVerify, YouTube | Write testbench for adder |
| 18 | Response Checking, Waveforms | 2-3 hrs | ChipVerify, EDA Playground | Analyze waveforms, verify outputs |

**Testbench Template:**
```verilog
module tb_adder;
  // Declare signals
  reg [3:0] a, b;
  reg cin;
  wire [3:0] sum;
  wire cout;
  
  // Instantiate DUT
  adder dut(a, b, cin, sum, cout);
  
  // Stimulus
  initial begin
    a = 4'h5; b = 4'h3; cin = 0;
    #10 check_output(4'h8, 0);
    
    a = 4'hF; b = 4'hF; cin = 1;
    #10 check_output(4'hF, 1);
    
    $finish;
  end
  
  // Check task
  task check_output(input [3:0] exp_sum, input exp_cout);
    if (sum !== exp_sum || cout !== exp_cout)
      $display("FAIL: got %h,%b, expected %h,%b", sum, cout, exp_sum, exp_cout);
    else
      $display("PASS");
  endtask
endmodule
```

---

#### **Week 19-20: Assertions & Coverage (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 19 | SystemVerilog Assertions, Properties | 2-3 hrs | ChipVerify, YouTube | Write 10 SVA assertions |
| 20 | Coverage-Driven Verification, Best Practices | 2-3 hrs | Tessolve PDF, ChipVerify | Create coverage model |

**Resources:**
- 📄 **Tessolve Intro to DV**: Free PDF
  - Link: https://www.tessolve.com/
- 🎥 **YouTube**: "SystemVerilog Assertions Tutorial"
  - Search for beginner SVA series

**SVA Example:**
```systemverilog
// Immediate assertion
always @(posedge clk) begin
  assert (counter <= 255) 
    else $error("Counter overflow!");
end

// Concurrent assertion
property counter_increment;
  @(posedge clk) (enable) |-> (counter == $past(counter) + 1);
endproperty

assert property(counter_increment);
```

---

#### **Week 21-22: Mini-Project 5 Completion (20 hours)**

**Mini-Project 5: Comprehensive Pipeline CPU Testbench**

**Deliverables:**
1. **Test Plan** (2-3 pages)
   - Test scenarios for all instruction types
   - Hazard test cases
   - Edge cases

2. **Testbench Code** (~300 lines SystemVerilog)
   - Stimulus generation module
   - Response checking module
   - Assertion definitions
   - Coverage collection

3. **Test Programs** (4 assembly programs)
   - Program 1: Basic arithmetic (10 instructions)
   - Program 2: Memory operations (10 instructions)
   - Program 3: Branches (10 instructions)
   - Program 4: Hazard induction (10 instructions)

4. **Results**
   - Waveform screenshots
   - Coverage report (target >90%)
   - Pass/fail summary

---

### MONTHS 7-8: AI/ML FUNDAMENTALS

**Goal:** Understand AI/ML concepts and train your first network

#### **Week 23-24: AI Concepts (No Coding) (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 23 | What is AI, ML vs DL, Supervised Learning | 2-3 hrs | Elements of AI, YouTube | Create AI concept mind-map |
| 24 | Neural Networks, Activation, Forward Pass | 2-3 hrs | 3Blue1Brown, Google ML Course | Explain forward propagation |

**Free Courses:**
- 🎓 **"Elements of AI"** by University of Helsinki
  - Link: https://www.elementsofai.com/
  - Duration: 4 hours, no coding
  - Perfect for conceptual foundation

- 🎥 **3Blue1Brown Neural Networks Series** (YouTube)
  - Link: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
  - 4 videos, very visual (1.5 hours)
  - Best intuition about how NNs work

- 🎓 **Google ML Crash Course**
  - Link: https://developers.google.com/machine-learning/crash-course
  - Interactive, practical (15 hours)
  - Can complete in 2 weeks if focused

---

#### **Week 25-26: Deep Learning Basics (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 25 | Backpropagation, CNNs, RNNs | 2-3 hrs | YouTube, Papers | Understand gradient flow |
| 26 | Transformers, Attention, Modern Architectures | 2-3 hrs | YouTube, OWASP | Architecture comparison table |

**Key Papers to Read:**
- "Attention Is All You Need" (Transformers)
  - Free: https://arxiv.org/abs/1706.03762
  - Skim sections 1-3 for understanding

---

#### **Week 27-28: Python & Neural Networks (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 27 | Python, NumPy, Data Handling | 2-3 hrs | NumPy tutorial, Codecademy | Write NumPy matrix operations |
| 28 | Neural Network from Scratch | 2-3 hrs | GitHub: Microsoft AI for Beginners | Implement simple 2-layer NN |

**Setup:**
```bash
pip install numpy matplotlib pandas jupyter scikit-learn
```

**Code Snippet (Forward Pass):**
```python
import numpy as np

class SimpleNN:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        # Hidden layer
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        
        # Output layer
        Z2 = A1 @ self.W2 + self.b2
        return Z2
```

---

#### **Week 29-30: Mini-Project 6 (20 hours)**

**Mini-Project 6: Train Neural Network on MNIST**

**Phases:**
1. **Data Loading**: Load MNIST, normalize, split
2. **Model**: Build 2-layer NN from scratch
3. **Training**: 10 epochs, batch size 128
4. **Profiling**: Measure time, memory per epoch
5. **Analysis**: Compare batch sizes 32, 128, 512

**Deliverables:**
- Jupyter notebook with:
  - Model code
  - Training loop
  - Evaluation
  - Plots of loss, accuracy, memory usage
- Report (2 pages):
  - Architecture description
  - Results (accuracy, time, memory)
  - Hardware insights

**Success Metrics:**
- [ ] Accuracy > 95% on MNIST
- [ ] Training time measured
- [ ] Memory usage profiled
- [ ] Batch size trade-offs analyzed

---

### MONTHS 9-10: HARDWARE FOR AI

**Goal:** Understand GPU architecture and design systolic arrays

#### **Week 31-32: GPU Architecture (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 31 | GPU vs CPU, Warps, Tensor Cores | 2-3 hrs | NVIDIA Blogs, Papers | GPU architecture diagram |
| 32 | TPU, Systolic Arrays, Memory Hierarchy | 2-3 hrs | Google TPU Paper, YouTube | Compare TPU vs GPU |

**Key Resources:**
- 📄 **"In-Datacenter Performance Analysis of a Tensor Processing Unit"**
  - Free: https://arxiv.org/abs/1704.04760
  - Read sections 1-4 (1-2 hours)

- 🎥 **NVIDIA Blogs**: GPU Architecture Series
  - Link: https://blogs.nvidia.com/
  - Search: "How GPUs Work"

---

#### **Week 33-34: Systolic Arrays & Matrix Multiply (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 33 | Systolic Array Concept, PE Design | 2-3 hrs | YouTube, Papers | Design single PE |
| 34 | 4×4 / 8×8 Array, Data Flow, Optimization | 2-3 hrs | Verilog Design | Plan full array architecture |

**Systolic Array PE (Pseudo-code):**
```
For each cycle:
  output_a = input_a
  output_b = input_b
  accumulator += input_a × input_b
  output_accumulator = accumulator
```

---

#### **Week 35-36: Roofline Model & Performance (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 35 | Roofline Model, Arithmetic Intensity | 2-3 hrs | Roofline Paper, YouTube | Plot roofline for GPU |
| 36 | Bottleneck Analysis, Profiling | 2-3 hrs | PyTorch Profiler, NVIDIA Nsight | Profile ResNet on GPU |

**Roofline Formula:**
```
Attainable Performance = min(Peak FLOPs, Peak Bandwidth × Arithmetic Intensity)

Arithmetic Intensity (AI) = FLOPs / Bytes Transferred
AI = (2 × M × N × K) / (M×K + K×N + M×N)  [for matrix multiply]

For large matrices (K>>1):
AI ≈ 2×M×N×K / 2×K×(M+N) ≈ (M×N) / (M+N)
```

---

#### **Week 37-38: Mini-Projects 7 & 8 (40 hours)**

**Mini-Project 7: Profile ML Workload (Week 37)**

Use PyTorch Profiler to analyze ResNet training:
```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
data = torch.randn(128, 3, 224, 224)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(data)
    loss = output.sum()
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Deliverables:**
- Profiler output showing top operations
- Roofline position analysis
- Bottleneck identification (compute vs memory)

---

**Mini-Project 8: Systolic Array in Verilog (Weeks 35-38)**

**Design 4×4 Systolic Array:**

**Files:**
1. `processing_element.v` (~50 lines)
   - Single PE with multiply-accumulate
2. `systolic_array.v` (~100 lines)
   - 16 PEs arranged in 4×4
3. `array_controller.v` (~80 lines)
   - Control data flow
4. `systolic_system.v` (~150 lines)
   - System integration, memory interface
5. `tb_systolic_array.sv` (~200 lines)
   - Comprehensive testbench

**Total Verilog: ~600 lines**

**Test Case:**
```python
import numpy as np

# Verify against NumPy
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)
C_expected = A @ B

# Simulate in Verilog
# Load weights row by row, activations column by column
# Read results after 4 + 4 - 1 = 7 cycles

# Verify C_hardware == C_expected (within rounding)
```

**Deliverables:**
- Verilog source files
- SystemVerilog testbench
- Design documentation (5 pages)
- Waveform screenshots
- Correctness verification
- Throughput analysis

---

### MONTHS 11-12: AI HARDWARE VALIDATION

**Goal:** Design and validate a complete AI accelerator

#### **Week 39-40: MLPerf & System Testing (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 39 | MLPerf Benchmarks, Inference, Training | 2-3 hrs | MLPerf website, GitHub | Run MLPerf on available hardware |
| 40 | System-Level Testing, NIST Framework | 2-3 hrs | NIST TEVV, OWASP, Tessolve | Create test specification |

**MLPerf Resources:**
- 🌐 **MLPerf Website**: https://mlperf.org/
  - Benchmark specifications
  - Results database
  - Leaderboards

- 📦 **MLPerf GitHub**: https://github.com/mlcommons/
  - Reference implementations
  - Docker containers
  - Setup instructions

**NIST TEVV Domains:**
- Accuracy & Robustness
- Fairness & Bias
- Explainability & Transparency
- Security & Privacy

---

#### **Week 41-42: Accelerator Specification & Design (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 41 | Architecture Planning, ISA Definition | 2-3 hrs | Your design | Detailed architecture document |
| 42 | Datapath Design, Memory Interface | 2-3 hrs | Verilog Planning | Block diagram, control signals |

**Accelerator Specification Template:**

```
=== SimpleAI-8 Accelerator ===

Compute:
- Datapath: 8×8 Systolic Array (from Mini-Project 8)
- Peak Performance: 512 MACs/cycle
- Clock Frequency: 500 MHz
- Peak FLOPs: 256 GFLOPs (FP32)

Memory:
- Input Buffer: 16 KB (weight cache)
- Activation Buffer: 16 KB
- DRAM: 8 GB HBM
- Peak Bandwidth: 900 GB/s

Instructions:
- MatMul: Systolic array computation
- Load: From DRAM to buffers
- Store: From buffers to DRAM
- Activate: Apply ReLU/activation
- Move: Data transfers between buffers

Target Workloads:
- ResNet-50 inference
- BERT inference
- GPT inference (smaller models)
```

---

#### **Week 43-44: Implementation & Integration (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 43 | Verilog Implementation, Memory Controller | 2-3 hrs | EDA Playground, Vivado | Core modules (500+ lines) |
| 44 | Instruction Decoder, Output Formatter | 2-3 hrs | Verilog | Integration & simulation |

**Module Hierarchy:**
```
ai_accelerator.v (top)
├── instruction_decoder.v
├── systolic_array.v (from Mini-Proj 8)
├── memory_controller.v
├── dma_engine.v
├── output_formatter.v
└── control_unit.v
```

---

#### **Week 45-46: Testbench & Validation (20 hours)**

| Week | Topics | Daily Time | Key Resources | Deliverable |
|------|--------|-----------|----------------|-------------|
| 45 | Test Plan, Stimulus Generation | 2-3 hrs | Your design | Comprehensive testbench |
| 46 | Verification, Results, Documentation | 2-3 hrs | Simulation | Final project deliverables |

---

#### **Week 47-48: Mini-Project 9 (40 hours)**

**Mini-Project 9: Complete AI Accelerator Design & Validation**

### **Deliverables:**

**1. Architecture Document (10 pages)**
- System overview
- Datapath diagram
- Instruction set definition (with examples)
- Memory interface specification
- Control signals and state machines
- Performance analysis

**2. Verilog Implementation (~800-1000 lines)**
- `ai_accelerator.v` (top module)
- `instruction_decoder.v`
- `systolic_array.v` (from Mini-Proj 8)
- `memory_controller.v` (DRAM simulation)
- `control_unit.v`
- `output_formatter.v`

**3. Verification Suite (SystemVerilog)**
- `tb_ai_accelerator.sv` (~300 lines)
- Test programs (assembly)
- Coverage specification
- SVA properties (10+)

**4. Test Programs**
- Program 1: Simple matrix multiply (2×2, 4×4)
- Program 2: Conv2D simulation (via tiled matrix multiplies)
- Program 3: Memory bandwidth test
- Program 4: Stress test (corner cases)

**5. Results & Analysis (5 pages)**
- Correctness verification
- Coverage report (target >95%)
- Performance metrics:
  - Throughput (GFLOPs achieved)
  - Utilization (% of peak)
  - Memory efficiency
  - Power estimation
- Comparison with roofline model
- Bottleneck analysis

**6. Waveform Screenshots**
- Instruction execution trace
- Data flow through systolic array
- Memory access patterns

---

## MINI-PROJECTS SUMMARY

### **P1: 4-Bit Binary Adder (Weeks 1-4, ~15 hours)**
- **Deliverable:** Schematic + verification
- **Skills:** Boolean algebra, circuit design
- **Platform:** Paper design, simulation tools

### **P2: 8-Bit Up/Down Counter (Weeks 5-6, ~12 hours)**
- **Deliverable:** FSM + timing diagrams
- **Skills:** Sequential logic, state machines
- **Platform:** Paper design + simple simulation

### **P3: RISC-V Program (Weeks 9-10, ~8 hours)**
- **Deliverable:** Assembly code + test output
- **Skills:** ISA, assembly programming
- **Platform:** RISC-V simulator (spike or Rars)

### **P4: 5-Stage Pipeline CPU (Weeks 9-14, ~30 hours)**
- **Deliverable:** Verilog code (~400 lines) + testbench
- **Skills:** CPU design, pipelining, HDL
- **Platform:** EDA Playground or Verilator
- **Assessment:** Executes 15-instruction program, shows pipeline behavior

### **P5: Comprehensive Testbench (Weeks 15-22, ~30 hours)**
- **Deliverable:** SystemVerilog testbench + test plans
- **Skills:** Verification, assertions, coverage
- **Platform:** EDA Playground / Verilator
- **Assessment:** >90% coverage, >10 assertions, all tests pass

### **P6: Neural Network on MNIST (Weeks 27-30, ~20 hours)**
- **Deliverable:** Jupyter notebook + report
- **Skills:** Python, NumPy, ML fundamentals
- **Platform:** Jupyter, Python
- **Assessment:** >95% accuracy, profiled for time/memory

### **P7: ML Workload Profiling (Week 37, ~10 hours)**
- **Deliverable:** Profiler output + analysis
- **Skills:** Performance analysis, roofline modeling
- **Platform:** PyTorch Profiler, NVIDIA Nsight
- **Assessment:** Identified bottlenecks, roofline plotted

### **P8: Systolic Array (Weeks 33-38, ~40 hours)**
- **Deliverable:** Verilog (~600 lines) + testbench
- **Skills:** Hardware design, systolic architectures
- **Platform:** EDA Playground / Vivado
- **Assessment:** Verified correct, measured throughput

### **P9: AI Accelerator (Weeks 41-48, ~50 hours)**
- **Deliverable:** Complete system (~1000 lines Verilog) + documentation
- **Skills:** All previous skills integrated
- **Platform:** EDA Playground / Vivado + synthesis
- **Assessment:** Correct execution, >95% coverage, documented results

---

## RESOURCES & LINKS

### **Digital Electronics**
| Resource | Link | Type | Cost |
|----------|------|------|------|
| Khan Academy | https://www.khanacademy.org/ | Video | Free |
| CircuitBread | https://www.circuitbread.com/tutorials/logic-gates | Tutorial | Free |
| All About Circuits | https://www.allaboutcircuits.com/textbook/digital/ | eBook | Free |
| Nandland | https://www.youtube.com/c/nandland | YouTube | Free |

### **Computer Architecture**
| Resource | Link | Type | Cost |
|----------|------|------|------|
| MIT OCW 6.823 | https://ocw.mit.edu/courses/6-823-computer-system-architecture-fall-2005/ | Course | Free |
| NPTEL IIT Delhi | https://nptel.ac.in/ | Course | Free |
| RISC-V Spec | https://riscv.org/ | Specification | Free |
| Roofline Model | https://people.eecs.berkeley.edu/~sameh/SC06.pdf | Paper | Free |

### **Hardware Verification**
| Resource | Link | Type | Cost |
|----------|------|------|------|
| ChipVerify | https://www.chipverify.com/ | Tutorial | Free |
| EDA Playground | https://edaplayground.com/ | Tool | Free |
| HDLBits | https://hdlbits.com/ | Practice | Free |
| Tessolve DV | https://www.tessolve.com/ | PDF | Free |

### **AI/ML**
| Resource | Link | Type | Cost |
|----------|------|------|------|
| Elements of AI | https://www.elementsofai.com/ | Course | Free |
| 3Blue1Brown NN | https://www.youtube.com/watch?v=aircAruvnKk | YouTube | Free |
| Google ML Course | https://developers.google.com/machine-learning/crash-course | Course | Free |
| Microsoft AI | https://github.com/microsoft/AI-For-Beginners | GitHub | Free |

### **Hardware for AI**
| Resource | Link | Type | Cost |
|----------|------|------|------|
| NVIDIA Blogs | https://blogs.nvidia.com/ | Blog | Free |
| Google TPU Paper | https://arxiv.org/abs/1704.04760 | Paper | Free |
| MLPerf | https://mlperf.org/ | Benchmarks | Free |
| PyTorch Profiler | https://pytorch.org/tutorials/recipes/recipes/profiler.html | Tool | Free |

---

## SUCCESS METRICS

### By Month 2:
- [ ] Can design circuits from logic gates
- [ ] Understand combinational & sequential logic
- [ ] Completed P1 & P2 mini-projects
- [ ] Score: 80%+ on digital electronics quiz

### By Month 4:
- [ ] Understand CPU design & pipelines
- [ ] Can write RISC-V assembly
- [ ] Completed P3 & P4 mini-projects
- [ ] 5-stage CPU executes programs correctly

### By Month 6:
- [ ] Can write professional testbenches
- [ ] Understand assertions & coverage
- [ ] Completed P5 mini-project
- [ ] Testbench achieves >90% coverage

### By Month 8:
- [ ] Understand AI/ML fundamentals
- [ ] Can train neural networks
- [ ] Completed P6 mini-project
- [ ] MNIST model >95% accurate

### By Month 10:
- [ ] Understand GPU/accelerator architecture
- [ ] Can design systolic arrays
- [ ] Completed P7 & P8 mini-projects
- [ ] Systolic array verified correct

### By Month 12:
- [ ] Complete AI accelerator designed & verified
- [ ] Completed P9 mini-project
- [ ] Documentation complete
- [ ] Portfolio: 5+ projects on GitHub

---

## GETTING STARTED CHECKLIST

### **This Week (Week 1):**

- [ ] Create GitHub account (if don't have one)
- [ ] Create new repo: "AI-Hardware-Validation-Learning"
- [ ] Watch Khan Academy "Binary System" (15 min)
- [ ] Read CircuitBread "Logic Gates" intro (30 min)
- [ ] Try EDA Playground: write & run simple AND gate (30 min)
- [ ] Take notes in markdown for future reference

**Total Time: 2 hours**

### **Week 1-2:**

- [ ] Complete Khan Academy number systems (3 hours)
- [ ] Complete CircuitBread logic gates tutorial (2 hours)
- [ ] Watch Nandland digital logic videos (2 hours)
- [ ] Practice: Design truth tables for 8 gates (1 hour)
- [ ] Create study notes & flashcards (2 hours)

**Total Time: 10 hours**

### **Week 3-4:**

- [ ] Start Mini-Project 1: 4-bit adder
- [ ] Draw schematic on paper (2 hours)
- [ ] Verify with test cases (1 hour)
- [ ] Create GitHub gist with design (1 hour)

**Total Time: 4 hours + project time**

---

## FAQ

**Q: Do I need expensive tools?**  
A: No. EDA Playground, Vivado (free version), Quartus (free version), and all learning resources are free.

**Q: Should I learn Verilog or SystemVerilog first?**  
A: Start with **Verilog**. It's simpler and sufficient for Months 1-6. SystemVerilog in Month 5 (for testbenches).

**Q: Can I skip any phases?**  
A: Not recommended. Each phase builds on the previous. But you can compress time if already familiar with a topic.

**Q: How many hours/week am I expected to spend?**  
A: 10-15 hours/week. Flexible - can stretch to 18+ months if doing 8 hrs/week, or compress to 9 months at 15+ hrs/week.

**Q: Should I read papers?**  
A: Start in Month 9+. Key papers: Google TPU, Roofline Model, MLPerf specs.

**Q: Can I contribute to open source during this?**  
A: Yes! Starting Month 9+, contribute to:
- RISC-V projects
- Hardware verification tools
- MLPerf benchmarks

---

## CONTACT & SUPPORT

**Having questions?**
- GitHub Discussions in your learning repo
- Reddit: r/FPGA, r/MachineLearning, r/VLSI
- Communities: EDAplayground forums, Nandland YouTube comments

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Total Learning Hours:** 120-180 hours over 12 months  

**Good luck! 🚀**
