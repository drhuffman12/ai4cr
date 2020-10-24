# Yeah, technicallu not a spec, but let's roll with this for now ...
# require "./../../../../spec_helper"
require "./../../../../src/ai4cr.cr"
require "benchmark"
require "ascii_bar_charter"

width = 100
height = 100
structure = [width, height]

training_io_qty = 10000
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

def histogram(arr, precision = 0) # , keys = [] of Float64)
  h = Hash(Float64, Int32).new
  # keys.each {|k| h[k] = 0}
  arr.flatten.group_by { |v| v.round(precision) }
    .each { |elem| h[elem[0]] = elem[1].size }
  h.to_a.sort { |a, b| a[0] <=> b[0] }.to_h
end

# net_ls_relu
#   TOTALS:: min: -51.140556, max: 66.240067, avg: -0.018731, stddev: 11.34342128217379
#   HISTOGRAM:: {
#   -51.0 => 3, -43.0 => 2, -41.0 => 2, -40.0 => 2, -39.0 => 1, -38.0 => 2, -37.0 => 3, -36.0 => 8, -35.0 => 3, -34.0 => 8,
#   -33.0 => 3, -32.0 => 9, -31.0 => 6, -30.0 => 10, -29.0 => 10, -28.0 => 12, -27.0 => 18, -26.0 => 26, -25.0 => 30, -24.0 => 27,
#   -23.0 => 52, -22.0 => 62, -21.0 => 74, -20.0 => 63, -19.0 => 78, -18.0 => 77, -17.0 => 112, -16.0 => 123, -15.0 => 142, -14.0 => 169,
#   -13.0 => 177, -12.0 => 201, -11.0 => 193, -10.0 => 253, -9.0 => 250, -8.0 => 272, -7.0 => 297, -6.0 => 337, -5.0 => 331, -4.0 => 368,
#   -3.0 => 357, -2.0 => 353, -1.0 => 360, -0.0 => 390, 1.0 => 364, 2.0 => 335, 3.0 => 366, 4.0 => 322, 5.0 => 334, 6.0 => 298,
#   7.0 => 317, 8.0 => 282, 9.0 => 240, 10.0 => 235, 11.0 => 242, 12.0 => 188, 13.0 => 172, 14.0 => 179, 15.0 => 127, 16.0 => 126,
#   17.0 => 109, 18.0 => 97, 19.0 => 77, 20.0 => 53, 21.0 => 56, 22.0 => 58, 23.0 => 37, 24.0 => 28, 25.0 => 38, 26.0 => 30
#    27.0 => 14, 28.0 => 22, 29.0 => 10, 30.0 => 8, 31.0 => 10, 32.0 => 9, 33.0 => 9, 34.0 => 9, 35.0 => 4, 36.0 => 3,
#    37.0 => 3, 38.0 => 2, 39.0 => 2, 41.0 => 3, 44.0 => 1, 45.0 => 1, 46.0 => 1, 47.0 => 1, 52.0 => 1, 66.0 => 1
# }

def plot_errors(name, net)
  puts "\n--------\n"
  puts name

  min = 0.0
  max = 1.0
  precision = 2.to_i8
  in_bw = false
  prefixed = false
  reversed = false

  charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
  plot = charter.plot(net.error_distance_history, prefixed)

  puts "  plot: '#{plot}'"
  puts "  error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"

  puts "\n--------\n"
end

def plot_weights(name, weights, verbose = false)
  puts "\n--------\n"
  puts name

  min = -1.0
  max = 1.0
  precision = 3.to_i8
  in_bw = false
  prefixed = false
  inverted_colors = true

  char_box = '\u2588' # 'x' # '\u25A0'
  # bar_chars = 11.times.to_a.map{ '\u25A0' }


  bar_colors = [:red, :black, :dark_gray, :yellow, :light_gray, :white, :green]
  # bar_chars = bar_colors.size.times.to_a.map{ '\u25A0' }
  bar_chars = bar_colors.size.times.to_a.map{ char_box }

  charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: inverted_colors)

  weights_flattened = weights.flatten
  puts "  TOTALS:: min: #{weights_flattened.min.round(precision*2)}, max: #{weights_flattened.max.round(precision*2)}, avg: #{(1.0 * weights_flattened.sum / weights_flattened.size).round(precision*2)}, stddev: #{weights_flattened.standard_deviation}"
  puts "  HISTOGRAM:: #{histogram(weights_flattened)}"
  puts "  ROWS::"
  weights.each do |row|
    plot = charter.plot(row, prefixed)

    puts "  plot: '#{plot}', min: #{row.min.round(precision*2)}, max: #{row.max.round(precision*2)}, avg: #{(1.0 * row.sum / weights_flattened.size).round(precision*2)}, stddev: #{row.standard_deviation}"
    puts "  row: '#{row.map { |e| e.round(precision*2) }}'" if verbose
  end

  puts "\n--------\n"
end

puts "\n========\n"

Benchmark.ips do |x|
  x.report("Initializing Backpropagation") { Ai4cr::NeuralNetwork::Backpropagation.new(structure: structure) }
  x.report("Initializing MiniNet (PRELU)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU) }
  x.report("Initializing MiniNet (RELU)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU) }
  x.report("Initializing MiniNet (SIGMOID)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID) }
  x.report("Initializing MiniNet (TANH)") { Ai4cr::NeuralNetwork::Cmn::MiniNet.new(width: width, height: height, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH) }
end

puts "\n========\n"

Benchmark.ips do |x|
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

puts "Errors and Trained Weights:"

plot_errors("net_backprop", net_backprop)
plot_weights("net_backprop(last)", net_backprop.weights.first)

plot_errors("net_ls_prelu", net_ls_prelu)
plot_weights("net_ls_prelu", net_ls_prelu.weights)

plot_errors("net_ls_relu", net_ls_relu)
plot_weights("net_ls_relu", net_ls_relu.weights)

plot_errors("net_ls_sigmoid", net_ls_sigmoid)
plot_weights("net_ls_sigmoid", net_ls_sigmoid.weights)

plot_errors("net_ls_tanh", net_ls_tanh)
plot_weights("net_ls_tanh", net_ls_tanh.weights)

puts "\n--------\n"

height_considering_bias = height + 1

Benchmark.ips do |x|
  x.report("Array.new") { Array.new(height_considering_bias) { |i| i } }
  x.report("to_a") { (0..(height_considering_bias - 1)).to_a }
end