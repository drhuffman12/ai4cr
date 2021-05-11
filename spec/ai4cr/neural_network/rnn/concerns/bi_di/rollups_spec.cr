require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::Rollups do
  let(hidden_size_given) { 0 } # aka the default (which leads to an actual hidden size of '1')
  let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

  let(all_outputs_expected_before) {
    [
      [
        {
          :channel_sl_or_combo => [0.0, 0.0, 0.0],
        },
        {
          :channel_sl_or_combo => [0.0, 0.0, 0.0],
        },
      ],
      [
        {
          :channel_forward     => [0.0, 0.0, 0.0],
          :channel_backward    => [0.0, 0.0, 0.0],
          :channel_sl_or_combo => [0.0],
        },
        {
          :channel_forward     => [0.0, 0.0, 0.0],
          :channel_backward    => [0.0, 0.0, 0.0],
          :channel_sl_or_combo => [0.0],
        },
      ],
    ]
  }

  describe "#all_mini_net_outputs" do
    context "returns" do
      it "expected values" do
        expect(rnn_bi_di.all_mini_net_outputs).to eq(all_outputs_expected_before)
      end
    end
  end
end
