require "json"
require "../../../../src/ai4cr"
require "../../../../src/ascii_bar_charter"

results_folder = "spec_examples/support/neural_network/backpropagation/data"
structure = [2,6,4]
disable_bias = true
net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure: structure, learning_rate: rand, disable_bias: disable_bias, momentum: rand)

stats = net.training_stats(in_bw: true)
puts "BEFORE any .. training_stats: #{stats}"

File.write("#{results_folder}/net_small_example.new.json",net.to_json)
File.write("#{results_folder}/net_small_example.new_stats.json",stats.to_json)

training_sets = [
  # binary (2 bits) to 4 columns (for values 0..3)
  { inputs: [0,0], outputs: [1,0,0,0] },
  { inputs: [0,1], outputs: [0,1,0,0] },
  { inputs: [1,0], outputs: [0,0,1,0] },
  { inputs: [1,1], outputs: [0,0,0,1] },
]
training_sets_size = training_sets.size
training_rounds = 100
training_rounds.times do |i|
  # errors = Array(Float64).new
  # training_sets.shuffle.each do |training_set|
  training_sets.each do |training_set|
    # errors << 
    net.train(training_set[:inputs], training_set[:outputs])
  end
  # error_averages << (errors.sum) / training_sets_size
end

stats = net.training_stats(in_bw: true)
puts "AFTER some .. training_stats: #{stats}"
File.write("#{results_folder}/net_small_example.trained.json",net.to_json)
File.write("#{results_folder}/net_small_example.trained_stats.json",stats.to_json)
