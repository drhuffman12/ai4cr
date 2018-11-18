require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnn::Net do
  describe "with default params" do
    net = Ai4cr::NeuralNetwork::Rnn::Net.new
    
  end
end



# icr
# require "./src/ai4cr/neural_network/rnn/net"
# net = Ai4cr::NeuralNetwork::Rnn::Net.new
# puts net.pretty_inspect

# puts "net.input_state_range: #{net.input_state_range}"
# puts "net.hidden_state_range: #{net.hidden_state_range}"
# puts "net.output_state_range: #{net.output_state_range}"
# puts "net.nodes_out: #{net.nodes_out}"

# time_column_index = 0
# hidden_layer_index = 0
# w = net.init_weights_to_current_past(time_column_index, hidden_layer_index)

# w = net.init_weights_to_current_future(time_column_index, hidden_layer_index)

# w = net.init_network_weights

# puts "w: #{w.pretty_inspect}"
          
# /home/drhuffman/_dev_/github.com/drhuffman12/ai4cr_alt/src/ai4cr/neural_network/Rnn/net.cr    

# mkdir spec/ai4cr/neural_network/Rnn/net

# crystal spec spec/ai4cr/neural_network/Rnn/net_spec.rb


