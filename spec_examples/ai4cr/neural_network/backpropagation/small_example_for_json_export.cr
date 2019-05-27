require "json"
require "../../../../src/ai4cr"
require "../../../../src/ascii_bar_charter"

# training_sets = [
#   # binary (2 bits) to 4 columns (for values 0..3)
#   { inputs: [0,0], outputs: [1,0,0,0] },
#   { inputs: [0,1], outputs: [0,1,0,0] },
#   { inputs: [1,0], outputs: [0,0,1,0] },
#   { inputs: [1,1], outputs: [0,0,0,1] },
# ]
# structure = [2,6,4]

training_sets = [
  # 4 columns (for values 0..3) to binary (2 bits)
  { outputs: [0,0], inputs: [1,0,0,0] },
  { outputs: [0,1], inputs: [0,1,0,0] },
  { outputs: [1,0], inputs: [0,0,1,0] },
  { outputs: [1,1], inputs: [0,0,0,1] },
]
structure = [4,6,2]

results_folder = "spec_examples/support/neural_network/backpropagation/data"

disable_bias = true
net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure: structure, learning_rate: rand, disable_bias: disable_bias, momentum: rand)

stats = net.training_stats(in_bw: true)
puts "BEFORE any .. training_stats: #{stats}"

File.write("#{results_folder}/net_small_example.new.json",net.to_json)
File.write("#{results_folder}/net_small_example.new.stats.json",stats.to_json)
File.write("#{results_folder}/net_small_example.new.state.json",net.state.to_json)
File.write("#{results_folder}/net_small_example.new.state.config.json",net.state.config.to_json)

training_sets_size = training_sets.size
training_rounds = 100
training_rounds.times do |i|
  # training_sets.shuffle.each do |training_set|
  training_sets.each do |training_set|
    net.train(training_set[:inputs], training_set[:outputs])
  end
end

stats = net.training_stats(in_bw: true)
puts "AFTER some .. training_stats: #{stats}"

File.write("#{results_folder}/net_small_example.trained.json",net.to_json)
File.write("#{results_folder}/net_small_example.trained.stats.json",stats.to_json)
File.write("#{results_folder}/net_small_example.trained.state.json",net.state.to_json)
File.write("#{results_folder}/net_small_example.trained.state.config.json",net.state.config.to_json)
