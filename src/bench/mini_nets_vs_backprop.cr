
require "option_parser"
require "benchmark"
require "ascii_bar_charter"

# require "../ai4cr/*"
# require "../ai4cr.cr"
require "../ai4cr.cr"

# require "../../spec/spec_helper"
# require "spec/test_helper"
# require "../../spec/spec_helper"
require "../../spec_examples/support/neural_network/data/*"

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

in_bw = false

OptionParser.parse do |parser|
  parser.banner = "Usage: mini_nets_vs_backprop [arguments]\n" +
    "Build: crystal build --release src/bench/mini_nets_vs_backprop.cr"
  parser.on("-b", "--in_bw", "Charts are in black and white") { in_bw = true }
  parser.on("-h", "--help", "Show this help") { puts parser }
end

struct Ai4cr::NeuralNetwork::Backpropagation
  def skipped_training_history
    [false]
  end
end

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
  { ins_arr: [tr_input, tr_with_noise, tr_with_base_noise], outs: is_a_triangle },
  { ins_arr: [sq_input, sq_with_noise, sq_with_base_noise], outs: is_a_square},
  { ins_arr: [cr_input, cr_with_noise, cr_with_base_noise], outs: is_a_cross}
]
ios_first = ios_list.first
ins_size = ios_first[:ins_arr].first.size
outs_size = ios_first[:outs].size
qty_loops = 100

min = 0.0
max = 1.0
precision = 2.to_i8
# in_bw = true
# in_bw = false
prefixed = false
# reversed = false
charter_high_is_red = AsciiBarCharter.new(min, max, precision, in_bw, reversed = false)
charter_high_is_blue = AsciiBarCharter.new(min, max, precision, in_bw, reversed = true)

def train(net, ios_list, qty_loops)
  # ins, outs
  qty_10_percent = qty_loops // 10
  (0..qty_loops).to_a.each do |i|
    ios = ios_list.sample
    ins = ios[:ins_arr].first # first is for training; all are for guessing
    outs = ios[:outs]
    net.train(ins, outs)
    # puts net.error_distance_history
    net.step_calculate_error_distance_history if i % qty_10_percent == 0 # if (i % ten_percent == 0)
  end
  net.error_distance_history
end

def array_mismatch_percentage(expected, actual)
  max = expected.size
  missed_qty = expected.map_with_index { |elem, i| elem == actual[i] ? 0 : 1 }.sum
  1.0 * missed_qty / max
end

# def array_mismatch_any?(expected, actual)
#   expected.map_with_index { |elem, i| elem != actual[i] }.any?
# end

def eval(net, ios_list)
  # puts "eval(net, ios_list) ... start"
  wrong_answers = Array(Float64).new
  ios_list.each do |ios|
    outs = ios[:outs]
    ios[:ins_arr].each do |ins|
      net.eval(ins)
      # puts "#{net.class.name}: expected: #{outs}, guesses_best: #{net.guesses_best || "error"}"
      # wrong_answers << ((outs ==  net.guesses_best) ? 0.0 : 1.0)
      wrong_answers << array_mismatch_percentage(outs, net.guesses_best)
      # wrong_answers << (assert_approximate_equality_of_nested_list(outs, net.guesses_best) ? 0.0 : 1.0)
    end
  end
  # puts "eval(net, ios_list) ... wrong_answers: #{wrong_answers}"
  # puts "eval(net, ios_list) ... end"
  wrong_answers
end

def graph(ios_list, charter_high_is_red, charter_high_is_blue, net)

  net_set_types = if net.is_a?(Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain)
    net.net_set.map{|ns| ns.class.name.split("::").last}.join(",")
  else
    ""
  end

  puts "#{net.class.name} with structure of #{net.structure} #{net_set_types}:"

  puts "\n--------\n"

  puts "  Training Error Rates:"
  plot_error_rates = charter_high_is_red.plot(net.error_distance_history, false) # false i.e.: NOT prefixed
  puts "    plot: '#{plot_error_rates}'"
  puts "    error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"

  puts "  Guessed-Wrong Percentages:"
  wrong_percentages = eval(net, ios_list)
  plot_wrong_percentages = charter_high_is_red.plot(wrong_percentages, false) # false i.e.: NOT prefixed

  puts "    plot: '#{plot_wrong_percentages}'"
  puts "    wrong_percentages: '#{wrong_percentages.map { |e| e.round(6) }}'"

  puts "  **** Guessed-Correct Percentages: ****"
  correct_percentages = wrong_percentages.map { |e| 1.0 - e }
  plot_correct_percentages = charter_high_is_blue.plot(correct_percentages, false)
  puts "    plot: '#{plot_correct_percentages}'"
  puts "    correct_percentages: '#{correct_percentages.map { |e| e.round(6) }}'"

  puts "  Skipped Training:"
  skipped_training_history_i = net.skipped_training_history.map{|h| h ? 1 : 0}
  plot_skip_rates = charter_high_is_blue.plot(skipped_training_history_i, false)
  puts "    plot: '#{plot_skip_rates}'"
  puts "    skipped_training_history: '#{skipped_training_history_i}'"

  puts "\n--------\n"
end

################################################################
puts "TRAINING #{qty_loops} times:"
################################################################

height = ins_size
width = outs_size

def bench_train_no_hidden(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, width)
  structure = [height, width]
  net_bp = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)
  net_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: height, width: width)
  net_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: height, width: width)
  net_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: height, width: width)

  puts "\n--------\n"
  Benchmark.ips do |x|
    x.report("Training of Backpropagation w/ structure of #{structure}") do
      train(net_bp, ios_list, qty_loops)
    end
    
    x.report("Training of MiniNet::Tanh w/ structure of #{structure}") do
      train(net_tanh, ios_list, qty_loops)
    end
    
    x.report("Training of MiniNet::Sigmoid w/ structure of #{structure}") do
      train(net_sigm, ios_list, qty_loops)
    end
    
    x.report("Training of MiniNet::Relu w/ structure of #{structure}") do
      train(net_relu, ios_list, qty_loops)
    end
  end

  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_bp)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_tanh)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_sigm)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_relu)
end

def bench_train_hidden1(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden, width)
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

  puts "\n--------\n"
  Benchmark.ips do |x|
    x.report("Training of Backpropagation w/ structure of #{structure}") do
      train(net_bp, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Tanh)") do
      train(cns_tanh_tanh, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid)") do
      train(cns_sigm_sigm, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Relu)") do
      train(cns_relu_relu, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Sigmoid)") do
      train(cns_relu_sigm, ios_list, qty_loops)
    end
  end

  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_bp)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_tanh_tanh)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_sigm_sigm)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_relu_relu)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_relu_sigm)   
end

def bench_train_hidden2(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden1, hidden2, width)
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
  


  net0_234_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: 2, width: 3)
  net1_234_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: 3, width: 4)
  net2_234_Sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: 4, width: 5)
  arr_234 = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
  arr_234 << net0_234_tanh
  arr_234 << net1_234_relu
  arr_234 << net2_234_Sigm
  cns_234 = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr_234)
  File.write("tmp/cns_234.json", cns_234.to_pretty_json(indent: "  "))
  # cns_234b = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.from_json(File.read("tmp/cns_234.json"))
  # File.write("tmp/cns_234b.json", cns_234b.to_pretty_json(indent: "  "))
  
  puts "\n--------\n"
  Benchmark.ips do |x|
    x.report("Training of Backpropagation w/ structure of #{structure}") do
      train(net_bp, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Tanh, Tanh)") do
      train(cns_tanh_tanh_tanh, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid, Sigmoid)") do
      train(cns_sigm_sigm_sigm, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Relu, Relu)") do
      train(cns_relu_relu_relu, ios_list, qty_loops)
    end
    
    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Relu, Sigmoid)") do
      train(cns_tanh_relu_sigm, ios_list, qty_loops)
    end
  end
  
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_bp)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_tanh_tanh_tanh)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_sigm_sigm_sigm)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_relu_relu_relu)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_tanh_relu_sigm)
end

## BENCHMARK 1a
# One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
# 
# width = 100
# height = 100
bench_train_no_hidden(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, width)

# ## BENCHMARK 1b
# # One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# # * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
# # 
# # width = 1000
# # height = 1000
# bench_train_no_hidden(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, width)

## BENCHMARK 2a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 100
hidden = 100
# height = 100
bench_train_hidden1(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden, width)

## BENCHMARK 2b
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 1000
hidden = 1000
# height = 1000
bench_train_hidden1(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden, width)

## BENCHMARK 3a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 100
hidden1 = 100
hidden2 = 100
# height = 100
bench_train_hidden2(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden1, hidden2, width)

## BENCHMARK 3b
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
# 
# width = 1000
hidden1 = 1000
hidden2 = 1000
# height = 1000
bench_train_hidden2(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden1, hidden2, width)

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

puts "\n--------\n"
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

puts "\n--------\n"
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

puts "\n--------\n"
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

puts "\n--------\n"
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

puts "\n--------\n"
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

puts "\n--------\n"
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
