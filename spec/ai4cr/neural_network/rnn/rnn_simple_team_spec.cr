require "./../../../spec_helper"
require "./../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeam do
  context "Team of RNN's Contained Mini Nets" do
    describe "#initialize" do
      let(team_size) { 2 }

      let(rnn_simple_team) {
        Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeam.new(
          team_size: team_size
        )
      }

      it "just some debugging" do # TODO: REMOVE before merging!
        # puts rnn_simple_team.to_pretty_json

        # rnn_simple_team.synaptic_layer_indexes.map do |li|
        #   rnn_simple_team.time_col_indexes.map do |ti|
        #     debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_simple_team.node_input_sizes[li][ti]}
        #     # puts debug_info.to_json
        #   end
        # end
      end
    end
  end
end
