require "benchmark"
require "ascii_bar_charter"

# require "../ai4cr/*"
# require "../ai4cr.cr"
require "../ai4cr.cr"

# require "spec/test_helper"
# require "../../spec/spec_helper"
require "../../spec_examples/support/neural_network/data/training_patterns"
require "../../spec_examples/support/neural_network/data/patterns_with_noise"
require "../../spec_examples/support/neural_network/data/patterns_with_base_noise"

# USAGE:
# time crystal build --release src/bench/backpropogation_1_vs_2.cr -o bin/bench/backpropogation_1_vs_2
# real    0m8.305s
# user    0m8.625s
# sys     0m0.167s
#
# time bin/bench/backpropogation_1_vs_2
# mkdir -p tmp/bench
# valgrind --tool=callgrind --inclusive=yes --tree=both --auto=yes --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation_1_vs_2.callgrind.out bin/bench/backpropogation_1_vs_2
# valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation_1_vs_2.callgrind.out bin/bench/backpropogation_1_vs_2

is_a_triangle = [1.0, 0.0, 0.0]
is_a_square = [0.0, 1.0, 0.0]
is_a_cross = [0.0, 0.0, 1.0]

tr_input = TRIANGLE.flatten.map { |input| input.to_f / 5.0 }
sq_input = SQUARE.flatten.map { |input| input.to_f / 5.0 }
cr_input = CROSS.flatten.map { |input| input.to_f / 5.0 }

tr_with_noise = TRIANGLE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
sq_with_noise = SQUARE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
cr_with_noise = CROSS_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }

tr_with_base_noise = TRIANGLE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
sq_with_base_noise = SQUARE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
cr_with_base_noise = CROSS_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }

ios_list = [
  { ins: is_a_triangle, outs_arr: [tr_input, tr_with_noise, tr_with_base_noise]},
  { ins: is_a_square, outs_arr: [sq_input, sq_with_noise, sq_with_base_noise]},
  { ins: is_a_cross, outs_arr: [cr_input, cr_with_noise, cr_with_base_noise]}
]
ios_first = ios_list.first
ins_size = ios_first[:ins].size
outs_size = ios_first[:outs_arr].first.size
qty_loops = 100

min = -1.0
max = 1.0
precision = 2.to_i8
in_bw = false
prefixed = false
reversed = false

charter = AsciiBarCharter.new(min, max, precision, in_bw, reversed)

def train(net, ios_list, qty_loops)
  # ins, outs_arr
  ten_percent = qty_loops // 10
  (0..qty_loops).to_a.each do |i|
    ios = ios_list.sample
    ins = ios[:ins]
    outs = ios[:outs_arr].first # first is for training; all are for guessing
    net.train(ins, outs)
    net.step_calculate_error_distance_history if (i % ten_percent == 0)
  end
end

def graph(charter, net)
  plot = charter.plot(net.error_distance_history, false) # false i.e.: NOT prefixed
  puts "\n--------\n"
  puts "#{net.class.name} with structure of #{net.structure}:"
  puts "  plot: '#{plot}'"
  puts "  error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"
  puts "\n--------\n"
end

################################################################
puts "INITIALIZATION ONLY:"
################################################################

## BENCHMARK 1a
# One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
# 
width = 100
height = 100
structure = [height, width]

puts "\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of MiniNet::Sigmoid w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: width) }
  x.report("Initialization of MiniNet::Tanh w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: width) }
  x.report("Initialization of MiniNet::Relu w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: width) }
end

## BENCHMARK 1b
# One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
# 
width = 1000
height = 1000
structure = [height, width]

puts "\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of MiniNet::Sigmoid w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: width) }
  x.report("Initialization of MiniNet::Tanh w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: width) }
  x.report("Initialization of MiniNet::Relu w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: width) }
end

## BENCHMARK 2a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
width = 100
hidden = 100
height = 100
structure = [height, hidden, width]

puts "\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid)") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
end

## BENCHMARK 2b
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
width = 1000
hidden = 1000
height = 1000
structure = [height, hidden, width]

puts "\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid)") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    # arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
end

## BENCHMARK 3a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
width = 100
hidden1 = 100
hidden2 = 100
height = 100
structure = [height, hidden1, hidden2, width]

puts "\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid)") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
end

## BENCHMARK 3b
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
width = 1000
hidden1 = 1000
hidden2 = 1000
height = 1000
structure = [height, hidden1, hidden2, width]

puts "\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid)") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden1)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden1, width: hidden2)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden2, width: width)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << net0
    arr << net1
    arr << net2
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
  end
end

################################################################
puts "TRAINING #{qty_loops} times:"
################################################################

height = ins_size
width = outs_size

def bench_train_no_hidden(ios_list, qty_loops, charter, height, width)
  structure = [height, width]
  net_bp = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)
  net_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: width)
  net_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: width)
  net_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: width)

  puts "\n"
  Benchmark.ips do |x|
    x.report("Training of Backpropagation w/ structure of #{structure}") do
      train(net_bp, ios_list, qty_loops)
    end
    graph(charter, net_bp)
    
    x.report("Training of MiniNet::Tanh w/ structure of #{structure}") do
      train(net_tanh, ios_list, qty_loops)
    end
    graph(charter, net_tanh)
    
    x.report("Training of MiniNet::Sigmoid w/ structure of #{structure}") do
      train(net_sigm, ios_list, qty_loops)
    end
    graph(charter, net_sigm)
    
    x.report("Training of MiniNet::Relu w/ structure of #{structure}") do
      train(net_relu, ios_list, qty_loops)
    end
    graph(charter, net_relu)
  end
end

def bench_train_hidden1(ios_list, qty_loops, charter, height, hidden, width)
  structure = [height, hidden, width]

  net_bp = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)

  net0_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden)
  net1_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_tanh
  arr << net1_tanh
  cns_tanh_tanh = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  net0_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: hidden)
  net1_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_sigm
  arr << net1_sigm
  cns_sigm_sigm = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  net0_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden)
  net1_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_relu
  arr << net1_relu
  cns_relu_relu = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  net0_relu_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden)
  net1_relu_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_relu_sigm
  arr << net1_relu_sigm
  cns_relu_sigm = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  puts "\n"
  Benchmark.ips do |x|
    x.report("Training of Backpropagation w/ structure of #{structure}") do
      train(net_bp, ios_list, qty_loops)
    end
    graph(charter, net_bp)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Tanh)") do
      train(cns_tanh_tanh, ios_list, qty_loops)
    end
    graph(charter, cns_tanh_tanh)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid)") do
      train(cns_sigm_sigm, ios_list, qty_loops)
    end
    graph(charter, cns_sigm_sigm)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Relu)") do
      train(cns_relu_relu, ios_list, qty_loops)
    end
    graph(charter, cns_relu_relu)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Sigmoid)") do
      train(cns_relu_sigm, ios_list, qty_loops)
    end
    graph(charter, cns_relu_sigm)
  end
end

def bench_train_hidden2(ios_list, qty_loops, charter, height, hidden1, hidden2, width)
  structure = [height, hidden1, hidden2, width]

  net_bp = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)

  net0_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden1)
  net1_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden1, width: hidden2)
  net2_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: hidden2, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_tanh
  arr << net1_tanh
  arr << net2_tanh
  cns_tanh_tanh_tanh = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  net0_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: hidden1)
  net1_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden1, width: hidden2)
  net2_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden2, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_sigm
  arr << net1_sigm
  arr << net2_sigm
  cns_sigm_sigm_sigm = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  net0_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: hidden1)
  net1_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden1, width: hidden2)
  net2_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden2, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_relu
  arr << net1_relu
  arr << net2_relu
  cns_relu_relu_relu = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  net0_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: hidden1)
  net1_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: hidden1, width: hidden2)
  net2_Sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: hidden2, width: width)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr << net0_tanh
  arr << net1_relu
  arr << net2_Sigm
  cns_tanh_relu_sigm = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

  puts "\n"
  Benchmark.ips do |x|
    x.report("Training of Backpropagation w/ structure of #{structure}") do
      train(net_bp, ios_list, qty_loops)
    end
    graph(charter, net_bp)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Tanh, Tanh)") do
      train(cns_tanh_tanh_tanh, ios_list, qty_loops)
    end
    graph(charter, cns_tanh_tanh_tanh)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid, Sigmoid)") do
      train(cns_sigm_sigm_sigm, ios_list, qty_loops)
    end
    graph(charter, cns_sigm_sigm_sigm)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Relu, Relu)") do
      train(cns_relu_relu_relu, ios_list, qty_loops)
    end
    graph(charter, cns_relu_relu_relu)
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Relu, Sigmoid)") do
      train(cns_tanh_relu_sigm, ios_list, qty_loops)
    end
    graph(charter, cns_tanh_relu_sigm)
  end
end

## BENCHMARK 1a
# One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
# 
# width = 100
# height = 100
bench_train_no_hidden(ios_list, qty_loops, charter, height, width)

# ## BENCHMARK 1b
# # One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# # * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
# # 
# # width = 1000
# # height = 1000
# bench_train_no_hidden(ios_list, qty_loops, charter, height, width)

## BENCHMARK 2a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 100
hidden = 100
# height = 100
bench_train_hidden1(ios_list, qty_loops, charter, height, hidden, width)

## BENCHMARK 2b
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 1000
hidden = 1000
# height = 1000
bench_train_hidden1(ios_list, qty_loops, charter, height, hidden, width)

## BENCHMARK 3a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 100
hidden1 = 100
hidden2 = 100
# height = 100
bench_train_hidden2(ios_list, qty_loops, charter, height, hidden1, hidden2, width)

## BENCHMARK 3b
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 1000
hidden1 = 1000
hidden2 = 1000
# height = 1000
bench_train_hidden2(ios_list, qty_loops, charter, height, hidden1, hidden2, width)
