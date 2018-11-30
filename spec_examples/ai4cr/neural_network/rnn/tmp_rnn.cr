
# require "./../../../../src/ai4cr/neural_network/rnn/node/input"
# node_input = Ai4cr::NeuralNetwork::Rnn::NodeSet::Input.new

# puts "initial node_input:"
# puts node_input.pretty_inspect(width: 120)

# require "./../../../../src/ai4cr/neural_network/rnn/node/trainable"
# node_trainable = Ai4cr::NeuralNetwork::Rnn::NodeSet::Trainable.new

# puts "initial node_trainable:"
# puts node_trainable.pretty_inspect(width: 120)

# require "./../../../../src/ai4cr/neural_network/rnn/node/hidden"
# node_hidden = Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden.new

# puts "initial node_hidden:"
# puts node_hidden.pretty_inspect(width: 120)

# require "./../../../../src/ai4cr/neural_network/rnn/node/output"
# node_output = Ai4cr::NeuralNetwork::Rnn::NodeSet::Output.new

# puts "initial node_output:"
# puts node_output.pretty_inspect(width: 120)

require "./../../../../src/ai4cr/neural_network/rnn/net"
net = Ai4cr::NeuralNetwork::Rnn::Net.new

puts "initial net:"
puts net.pretty_inspect(width: 120)

# # hidden_layer_index = 0
# # time_column_index = 0

# # den_maps = net.init_dendrite_mappings(hidden_layer_index, time_column_index)
# # puts "1st layer den_maps:"
# # puts den_maps.pretty_inspect(width: 120)

# # sums = net.forward_sum_local(hidden_layer_index, time_column_index)
# # puts "1st layer sumed:"
# # puts sums.pretty_inspect(width: 120)

# # outs = net.forward_propogate_local(hidden_layer_index, time_column_index)
# # puts "1st layer propagated:"
# # puts outs.pretty_inspect(width: 120)

