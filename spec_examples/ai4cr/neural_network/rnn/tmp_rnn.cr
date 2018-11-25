
# require "./../../../../src/ai4cr/neural_network/rnn/node/input"
# node_input = Ai4cr::NeuralNetwork::Rnn::Node::Input.new

# puts "initial node_input:"
# puts node_input.pretty_inspect

# require "./../../../../src/ai4cr/neural_network/rnn/node/trainable"
# node_trainable = Ai4cr::NeuralNetwork::Rnn::Node::Trainable.new

# puts "initial node_trainable:"
# puts node_trainable.pretty_inspect

# require "./../../../../src/ai4cr/neural_network/rnn/node/hidden"
# node_hidden = Ai4cr::NeuralNetwork::Rnn::Node::Hidden.new

# puts "initial node_hidden:"
# puts node_hidden.pretty_inspect

# require "./../../../../src/ai4cr/neural_network/rnn/node/output"
# node_output = Ai4cr::NeuralNetwork::Rnn::Node::Output.new

# puts "initial node_output:"
# puts node_output.pretty_inspect

require "./../../../../src/ai4cr/neural_network/rnn/net"
net = Ai4cr::NeuralNetwork::Rnn::Net.new

puts "initial net:"
puts net.pretty_inspect

# # hidden_layer_index = 0
# # time_column_index = 0

# # den_maps = net.init_dendrite_mappings(hidden_layer_index, time_column_index)
# # puts "1st layer den_maps:"
# # puts den_maps.pretty_inspect

# # sums = net.forward_sum_local(hidden_layer_index, time_column_index)
# # puts "1st layer sumed:"
# # puts sums.pretty_inspect

# # outs = net.forward_propogate_local(hidden_layer_index, time_column_index)
# # puts "1st layer propagated:"
# # puts outs.pretty_inspect

