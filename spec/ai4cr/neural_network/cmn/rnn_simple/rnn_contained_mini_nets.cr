require "./../../../../spec_helper"
require "./../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnSimple do
  context "RNN Contained Mini Nets" do
    describe "#initialize" do
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

      it "just some debugging" do # TODO: REMOVE before merging!
        puts rnn_simple.to_pretty_json

        # rnn_simple.nodal_layer_indexes.map do |li|
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            # debug_info = {"li": li, "ti": ti, "rnn_simple.node_input_sizes[li][ti]": rnn_simple.node_input_sizes[li][ti]}
            debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple.node_input_sizes[li][ti]}
            puts debug_info.to_json
          end
        end
      end
    end
  end
end
