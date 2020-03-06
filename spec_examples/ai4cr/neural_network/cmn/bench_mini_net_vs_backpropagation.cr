# require "./../../../../spec_helper"
require "./../../../../src/ai4cr.cr"
require "benchmark"
require "ascii_bar_charter"

width = 100
height = 100
structure = [width, height]

training_io_qty = 100
graph_sample_percent = training_io_qty // 20
training_io_indexes = training_io_qty.times.to_a

height_indexes = height.times.to_a
width_indexes = width.times.to_a

example_input_set = training_io_indexes.map { height_indexes.map { rand().to_f } }
example_output_set = training_io_indexes.map { width_indexes.map { rand().round.to_f } }

example_input_set_tanh = training_io_indexes.map { height_indexes.map { (rand()*2 - 1).to_f } }
example_output_set_tanh = training_io_indexes.map { width_indexes.map { (rand()*2 - 1).round.to_f } }

net_backprop = Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure)
net_ls_prelu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU)
net_ls_relu = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU)
net_ls_sigmoid = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
net_ls_tanh = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH)

def ascii_plot(name, net)
  puts "\n--------\n"
  puts name

  min = 0.0
  max = 1.0
  precision = 2.to_i8
  in_bw = false
  prefixed = false
  reversed = false

  charter = AsciiBarCharter.new(min, max, precision, in_bw, reversed)
  plot = charter.plot(net.error_distance_history, prefixed)

  puts "  plot: '#{plot}'"
  puts "  error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"

  puts "\n--------\n"
end

puts "\n========\n"

results = Benchmark.ips do |x|
  x.report("Initializing Backpropagation") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initializing MiniNet (PRELU)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU) }
  x.report("Initializing MiniNet (RELU)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU) }
  x.report("Initializing MiniNet (SIGMOID)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID) }
  x.report("Initializing MiniNet (TANH)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH) }
end

puts "\n========\n"

results = Benchmark.ips do |x|
  x.report("Training (on random data) Backpropagation") do
    training_io_indexes.each do |i|
      net = net_backprop
      net.train(example_input_set[i], example_output_set[i])
      net.step_calculate_error_distance_history if i % graph_sample_percent == 0
    end
  end
  x.report("Training (on random data) MiniNet (PRELU)") do
    training_io_indexes.each do |i|
      net = net_ls_prelu
      net.train(example_input_set[i], example_output_set[i])
      net.step_calculate_error_distance_history if i % graph_sample_percent == 0
    end
  end
  x.report("Training (on random data) MiniNet (RELU)") do
    training_io_indexes.each do |i|
      net = net_ls_relu
      net.train(example_input_set[i], example_output_set[i])
      net.step_calculate_error_distance_history if i % graph_sample_percent == 0
    end
  end
  x.report("Training (on random data) MiniNet (SIGMOID)") do
    training_io_indexes.each do |i|
      net = net_ls_sigmoid
      net.train(example_input_set[i], example_output_set[i])
      net.step_calculate_error_distance_history if i % graph_sample_percent == 0
    end
  end
  x.report("Training (on random data) MiniNet (TANH)") do
    training_io_indexes.each do |i|
      net = net_ls_tanh
      net.train(example_input_set_tanh[i], example_output_set_tanh[i])
      net.step_calculate_error_distance_history if i % graph_sample_percent == 0
    end
  end
end

puts "\n========\n"

# training_io_indexes.each do |i|
#   describe "Compare guesses based on random value training at index #{i}" do
#     it "net_backprop" do
#       net = net_backprop
#       net.eval(example_input_set[i])
#       # assert_approximate_equality_of_nested_list net.guesses_best, example_output_set[i], 0.1
#       assert_equality_of_nested_list net.guesses_best, example_output_set[i]
#     end

#     it "net_ls_prelu" do
#       net = net_ls_prelu
#       net.eval(example_input_set[i])
#       # assert_approximate_equality_of_nested_list net.guesses_best, example_output_set[i], 0.1
#       assert_equality_of_nested_list net.guesses_best, example_output_set[i]
#     end

#     it "net_ls_relu" do
#       net = net_ls_relu
#       net.eval(example_input_set[i])
#       # assert_approximate_equality_of_nested_list net.guesses_best, example_output_set[i], 0.1
#       assert_equality_of_nested_list net.guesses_best, example_output_set[i]
#     end

#     it "net_ls_sigmoid" do
#       net = net_ls_sigmoid
#       net.eval(example_input_set[i])
#       # assert_approximate_equality_of_nested_list net.guesses_best, example_output_set[i], 0.1
#       assert_equality_of_nested_list net.guesses_best, example_output_set[i]
#     end

#     it "net_ls_tanh" do
#       net = net_ls_tanh
#       net.eval(example_input_set_tanh[i])
#       # assert_approximate_equality_of_nested_list net.guesses_best, example_output_set[i], 0.1
#       assert_equality_of_nested_list net.guesses_best, example_output_set_tanh[i]
#     end
#   end
# end

sleep 5

puts "\n========\n"

ascii_plot("net_backprop", net_backprop)
ascii_plot("net_ls_prelu", net_ls_prelu)
ascii_plot("net_ls_relu", net_ls_relu)
ascii_plot("net_ls_sigmoid", net_ls_sigmoid)
ascii_plot("net_ls_tanh", net_ls_tanh)

puts "\n========\n"

height_considering_bias = height + 1

results = Benchmark.ips do |x|
  x.report("Array.new") { Array.new(height_considering_bias) { |i| i } }
  x.report("to_a") { (0..(height_considering_bias - 1)).to_a }
end