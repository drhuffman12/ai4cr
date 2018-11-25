require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnn::Net do
  describe "#initialize" do
    
    describe "with default params" do
      net = Ai4cr::NeuralNetwork::Rnn::Net.new
      # default_nodes_hidden_channel = [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]

      
      # describe "sets expected quantity for instance variable" do
      #   it "@bias" do
      #     net.bias.should eq(true)
      #   end

      #   it "@deltas_hidden_combo" do
      #     net.deltas_hidden_combo.should eq(default_nodes_hidden_channel)
      #   end

      #   it "@deltas_hidden_future" do
      #     net.deltas_hidden_future.should eq(default_nodes_hidden_channel)
      #   end

      #   it "@deltas_hidden_local" do
      #     net.deltas_hidden_local.should eq(default_nodes_hidden_channel)
      #   end

      #   it "@deltas_hidden_past" do
      #     net.deltas_hidden_past.should eq(default_nodes_hidden_channel)
      #   end
    
      #   # TODO (?): the others
      # end
    end
  end
end

