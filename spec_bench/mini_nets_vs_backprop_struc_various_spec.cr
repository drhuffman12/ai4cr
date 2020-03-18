require "option_parser"
require "benchmark"
require "ascii_bar_charter"
require "./../src/ai4cr.cr"

require "./../spec_examples/support/neural_network/data/*"

include Ai4cr::NeuralNetwork::ChartingAndPlotting

# To dig deeper into performance refinement:
# crystal build --release src/mini_nets_vs_backprop
# mkdir -p tmp/
# valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/mini_nets_vs_backprop.out ./mini_nets_vs_backprop

in_bw = false

OptionParser.parse do |parser|
  parser.banner = "Usage: mini_nets_vs_backprop [arguments]\n" +
                  "Build: crystal build --release src/bench/mini_nets_vs_backprop.cr"
  parser.on("-b", "--in_bw", "Charts are in black and white") { in_bw = true }
  parser.on("-h", "--help", "Show this help") { puts parser }
end

# struct Ai4cr::NeuralNetwork::Backpropagation
#   def skipped_training_history
#     [false]
#   end
# end

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
  {ins_arr: [tr_input, tr_with_noise, tr_with_base_noise], outs: is_a_triangle},
  {ins_arr: [sq_input, sq_with_noise, sq_with_base_noise], outs: is_a_square},
  {ins_arr: [cr_input, cr_with_noise, cr_with_base_noise], outs: is_a_cross},
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
# prefixed = false
# inverted_colors = false
charter_high_is_red = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: false)
charter_high_is_blue = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: true)

def train(net, ios_list, qty_loops)
  # ins, outs
  qty_10_percent = qty_loops // 10
  (0..qty_loops).to_a.each do |i|
    ios = ios_list.sample
    ins = ios[:ins_arr].first # first is for training; all are for guessing
    outs = ios[:outs]
    # if net.class == Ai4cr::NeuralNetwork::Backpropagation
    #   net.train(ins, outs)
    # else
    #   net.train(ins, outs) # , [] of Float64, false)
    # end
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

def graph(ios_list, charter_high_is_red, charter_high_is_blue, net, initial_weights_per_layer)
  net_set_types = if net.is_a?(Ai4cr::NeuralNetwork::Cmn::Chain)
                    # net.mini_net_set.map { |mn| mn.class.name.split("::").last }.join(",")
                    net.mini_net_set.map { |mn| mn.learning_style }.join(",")
                  elsif net.is_a?(Ai4cr::NeuralNetwork::Cmn::MiniNet)
                    net.learning_style
                  else
                    "n/a"
                  end

  puts "#{net.class.name} with structure of #{net.structure} (#{net_set_types}):"

  puts "\n--------\n"

  puts "  Training Error Rates:"
  plot_error_rates = charter_high_is_red.plot(net.error_distance_history, false) # false i.e.: NOT prefixed
  puts "    plot: '#{plot_error_rates}'"
  puts "    error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"

  precision = 0.1
  # net_set_types =
  if net.is_a?(Ai4cr::NeuralNetwork::Cmn::Chain)
    net.mini_net_set.each_with_index do |mn, i|
      plot_histogram("    weight histogram [#{i}]", mn.weights, precision)
    end
  elsif net.is_a?(Ai4cr::NeuralNetwork::Cmn::MiniNet)
    plot_histogram("    weight histogram", net.weights, precision)
  else
    net.weights.each_with_index do |w, i|
      plot_histogram("    weight histogram [#{i}]", w, precision)
    end
  end
  initial_weights_per_layer.each_with_index do |w, i|
    plot_histogram("    initial_weights_per_layer histogram [#{i}]", w, precision)
  end

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
  net_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  net_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  net_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  net_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)

  # initial_weights_per_layer = [net_relu.weights]
  initial_weights_per_layer = [net_sigm.weights]

  initial_weights_per_layer.each_with_index do |weights, i|
    raise "Weight Mismatch re net_bp" if net_bp.weights[i].size != weights.size && net_bp.weights[i].flatten.size != weights.flatten.size
    raise "Weight Mismatch re net_tanh : i: #{i}, net_tanh.weights.size: #{net_tanh.weights.size}, weights.size: #{weights.size}, net_tanh.weights.flatten.size: #{net_tanh.weights.flatten.size}, weights.flatten.size: #{weights.flatten.size}" if net_tanh.weights.size != weights.size && net_tanh.weights.flatten.size != weights.flatten.size
    net_bp.weights[i] = weights.clone
    net_tanh.weights = weights.clone
    net_sigm.weights = weights.clone
    net_relu.weights = weights.clone
    net_prelu.weights = weights.clone
  end

  puts "\n--------\n"
  Benchmark.ips do |x|
    x.report("Training of Backpropagation w/ structure of #{structure}") do
      train(net_bp, ios_list, qty_loops)
    end

    x.report("Training of MiniNet (Tanh) w/ structure of #{structure}") do
      train(net_tanh, ios_list, qty_loops)
    end

    x.report("Training of MiniNet (Sigmoid) w/ structure of #{structure}") do
      train(net_sigm, ios_list, qty_loops)
    end

    x.report("Training of MiniNet (Relu) w/ structure of #{structure}") do
      train(net_relu, ios_list, qty_loops)
    end

    x.report("Training of MiniNet (Prelu) w/ structure of #{structure}") do
      train(net_prelu, ios_list, qty_loops)
    end
  end

  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_bp, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_tanh, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_sigm, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_relu, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, net_prelu, initial_weights_per_layer)
end

def bench_train_hidden1(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden, width)
  structure = [height, hidden, width]

  net_bp = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)

  net0_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  net1_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_tanh
  arr << net1_tanh
  cns_tanh_tanh = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  net1_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_sigm
  arr << net1_sigm
  cns_sigm_sigm = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  net1_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_relu
  arr << net1_relu
  cns_relu_relu = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
  net1_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_prelu
  arr << net1_prelu
  cns_prelu_prelu = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_relu_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  net1_relu_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_relu_sigm
  arr << net1_relu_sigm
  cns_relu_sigm = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  # initial_weights_per_layer = net_relu.mini_net_set.map_with_index do |mn, i|
  initial_weights_per_layer = cns_sigm_sigm.mini_net_set.map_with_index do |mn, i|
    net_bp.weights[i] = mn.weights
  end

  initial_weights_per_layer.each_with_index do |weights, i|
    raise "Weight Mismatch re net_bp" if net_bp.weights[i].size != weights.size && net_bp.weights[i].flatten.size != weights.flatten.size
    raise "Weight Mismatch re cns_tanh_tanh : i: #{i}, cns_tanh_tanh.mini_net_set[i].weights.size: #{cns_tanh_tanh.mini_net_set[i].weights.size}, weights.size: #{weights.size}, cns_tanh_tanh.mini_net_set[i].weights.flatten.size: #{cns_tanh_tanh.mini_net_set[i].weights.flatten.size}, weights.flatten.size: #{weights.flatten.size}" if cns_tanh_tanh.mini_net_set[i].weights.size != weights.size && cns_tanh_tanh.mini_net_set[i].weights.flatten.size != weights.flatten.size

    net_bp.weights[i] = weights.clone
    cns_tanh_tanh.mini_net_set[i].weights = weights.clone
    cns_sigm_sigm.mini_net_set[i].weights = weights.clone
    cns_relu_relu.mini_net_set[i].weights = weights.clone
    cns_prelu_prelu.mini_net_set[i].weights = weights.clone
    cns_relu_sigm.mini_net_set[i].weights = weights.clone
  end

  puts "\n--------\n"
  Benchmark.ips do |x|
    # x.report("Training of Backpropagation w/ structure of #{structure}") do
    #   train(net_bp, ios_list, qty_loops)
    # end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Tanh)") do
      train(cns_tanh_tanh, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid)") do
      train(cns_sigm_sigm, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Relu)") do
      train(cns_relu_relu, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Prelu, Prelu)") do
      train(cns_prelu_prelu, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Sigmoid)") do
      train(cns_relu_sigm, ios_list, qty_loops)
    end
  end

  # graph(ios_list, charter_high_is_red, charter_high_is_blue, net_bp, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_tanh_tanh, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_sigm_sigm, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_relu_relu, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_relu_sigm, initial_weights_per_layer)
end

def bench_train_hidden2(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden1, hidden2, width)
  structure = [height, hidden1, hidden2, width]

  net_bp = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)

  net0_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  net1_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  net2_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_tanh
  arr << net1_tanh
  arr << net2_tanh
  cns_tanh_tanh_tanh = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  net1_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  net2_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_sigm
  arr << net1_sigm
  arr << net2_sigm
  cns_sigm_sigm_sigm = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  net1_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  net2_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_relu
  arr << net1_relu
  arr << net2_relu
  cns_relu_relu_relu = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
  net1_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
  net2_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_prelu
  arr << net1_prelu
  arr << net2_prelu
  cns_prelu_prelu_prelu = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  net1_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  net2_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr << net0_tanh
  arr << net1_relu
  arr << net2_sigm
  cns_tanh_relu_sigm = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  net0_234_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 2, width: 3, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
  net1_234_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 3, width: 4, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
  net2_234_sigm = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 4, width: 5, bias_disabled: true)
  arr_234 = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  arr_234 << net0_234_tanh
  arr_234 << net1_234_relu
  arr_234 << net2_234_sigm
  cns_234 = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr_234)
  # File.write("tmp/cns_234.json", cns_234.to_pretty_json(indent: "  "))
  # cns_234b = Ai4cr::NeuralNetwork::Cmn::Chain.from_json(File.read("tmp/cns_234.json"))
  # File.write("tmp/cns_234b.json", cns_234b.to_pretty_json(indent: "  "))
  puts cns_234.to_pretty_json(indent: "  ")

  # initial_weights_per_layer = net_relu.mini_net_set.map_with_index do |mn, i|
  initial_weights_per_layer = cns_sigm_sigm_sigm.mini_net_set.map_with_index do |mn, i|
    net_bp.weights[i] = mn.weights
  end

  initial_weights_per_layer.each_with_index do |weights, i|
    raise "Weight Mismatch re net_bp" if net_bp.weights[i].size != weights.size && net_bp.weights[i].flatten.size != weights.flatten.size
    raise "Weight Mismatch re cns_tanh_tanh_tanh : i: #{i}, cns_tanh_tanh_tanh.mini_net_set[i].weights.size: #{cns_tanh_tanh_tanh.mini_net_set[i].weights.size}, weights.size: #{weights.size}, cns_tanh_tanh_tanh.mini_net_set[i].weights.flatten.size: #{cns_tanh_tanh_tanh.mini_net_set[i].weights.flatten.size}, weights.flatten.size: #{weights.flatten.size}" if cns_tanh_tanh_tanh.mini_net_set[i].weights.size != weights.size && cns_tanh_tanh_tanh.mini_net_set[i].weights.flatten.size != weights.flatten.size

    net_bp.weights[i] = weights.clone
    cns_tanh_tanh_tanh.mini_net_set[i].weights = weights.clone
    cns_sigm_sigm_sigm.mini_net_set[i].weights = weights.clone
    cns_relu_relu_relu.mini_net_set[i].weights = weights.clone
    cns_prelu_prelu_prelu.mini_net_set[i].weights = weights.clone
    cns_tanh_relu_sigm.mini_net_set[i].weights = weights.clone
  end

  puts "\n--------\n"
  Benchmark.ips do |x|
    # x.report("Training of Backpropagation w/ structure of #{structure}") do
    #   train(net_bp, ios_list, qty_loops)
    # end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Tanh, Tanh)") do
      train(cns_tanh_tanh_tanh, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Sigmoid, Sigmoid, Sigmoid)") do
      train(cns_sigm_sigm_sigm, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Relu, Relu)") do
      train(cns_relu_relu_relu, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Relu, Relu, Relu)") do
      train(cns_prelu_prelu_prelu, ios_list, qty_loops)
    end

    x.report("Training of ConnectedNetSet::Chain w/ structure of #{structure} (Tanh, Relu, Sigmoid)") do
      train(cns_tanh_relu_sigm, ios_list, qty_loops)
    end
  end

  # graph(ios_list, charter_high_is_red, charter_high_is_blue, net_bp, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_tanh_tanh_tanh, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_sigm_sigm_sigm, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_relu_relu_relu, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_prelu_prelu_prelu, initial_weights_per_layer)
  graph(ios_list, charter_high_is_red, charter_high_is_blue, cns_tanh_relu_sigm, initial_weights_per_layer)
end

# # BENCHMARK 1a
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

# # BENCHMARK 2a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
#
# width = 100
hidden = 100
# height = 100
bench_train_hidden1(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden, width)

# # BENCHMARK 2b
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
#
# width = 1000
hidden = 1000
# height = 1000
bench_train_hidden1(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden, width)

# # BENCHMARK 3a
# Two nets w/ the following 2 layers of weights and 3 layers of nodes with the following structure: [100,100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 hidden nodes
# * 2nd net's weights are for the 100 hidden nodes to the 100 output nodes
#
# width = 100
hidden1 = 100
hidden2 = 100
# height = 100
bench_train_hidden2(ios_list, qty_loops, charter_high_is_red, charter_high_is_blue, height, hidden1, hidden2, width)

# # BENCHMARK 3b
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

# # BENCHMARK 1a
# One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
#
width = 100
height = 100
structure = [height, width]

puts "\n--------\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of MiniNet (Sigmoid) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID) }
  x.report("Initialization of MiniNet (Tanh) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH) }
  x.report("Initialization of MiniNet (Relu) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU) }
  x.report("Initialization of MiniNet (Prelu) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU) }
end

# # BENCHMARK 1b
# One net w/ the following 1 layers of weights and 2 layers of nodes with the following structure: [100,100]
# * 1st net's weights are for the 100 inputs nodes to the 100 output nodes
#
width = 1000
height = 1000
structure = [height, width]

puts "\n--------\n"
Benchmark.ips do |x|
  x.report("Initialization of Backpropagation w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initialization of MiniNet (Sigmoid) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID) }
  x.report("Initialization of MiniNet (Tanh) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH) }
  x.report("Initialization of MiniNet (Relu) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU) }
  x.report("Initialization of MiniNet (Prelu) w/ structure of #{structure}") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: width, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU) }
end

# # BENCHMARK 2a
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
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Prelu, PrRelu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
end

# # BENCHMARK 2b
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
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    # arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Prelu, Prelu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    # arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
end

# # BENCHMARK 3a
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
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Prelu, Prelu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
end

# # BENCHMARK 3b
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
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Tanh, Tanh") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Relu, Relu, Relu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Prelu, Prelu, Prelu") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
  x.report("Initialization of ConnectedNetSet::Chain (Tanh, Relu, Sigmoid") do
    net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: height, width: hidden1, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)
    net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden1, width: hidden2, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
    net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden2, width: width, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
    arr << net0
    arr << net1
    arr << net2
    Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
  end
end
