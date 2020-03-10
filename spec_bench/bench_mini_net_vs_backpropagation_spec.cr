# Yeah, technicallu not a spec, but let's roll with this for now ...
# require "./../../../../spec_helper"
require "benchmark"
require "ascii_bar_charter"
require "./../src/ai4cr.cr"

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

def histogram(arr, precision = 0) # , keys = [] of Float64)
  h = Hash(Float64, Int32).new
  # keys.each {|k| h[k] = 0}
  arr.flatten.group_by { |v| v.round(precision) }
    .each { |elem| h[elem[0]] = elem[1].size }
  h.to_a.sort { |a, b| a[0] <=> b[0] }.to_h
end

def plot_errors(name, net)
  puts "\n--------\n"
  puts name

  min = 0.0
  max = 1.0
  precision = 2.to_i8
  in_bw = false
  prefixed = false
  inverted_colors = false

  charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: inverted_colors)
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

  charter = AsciiBarCharter.new(
    min: min, max: max, precision: precision, 

    # bar_chars: AsciiBarCharter::BAR_CHARS,
    # bar_colors: AsciiBarCharter::BAR_COLORS,
    bar_chars: bar_chars,

    # bar_chars: AsciiBarCharter::BAR_CHARS_ALT,
    # bar_colors: AsciiBarCharter::BAR_COLORS_ALT,
    bar_colors: bar_colors,

    in_bw: in_bw, inverted_colors: inverted_colors
  )

  weights_flattened = weights.flatten
  puts "  TOTALS:: min: #{weights_flattened.min.round(precision*2)}, max: #{weights_flattened.max.round(precision*2)}, avg: #{(1.0 * weights_flattened.sum / weights_flattened.size).round(precision*2)}, stddev: #{weights_flattened.standard_deviation}"
  # puts "  HISTOGRAM:: #{histogram(weights_flattened)}"
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
