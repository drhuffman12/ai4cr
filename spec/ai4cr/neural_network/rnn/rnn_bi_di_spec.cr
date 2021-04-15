require "./../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnBiDi do
  describe "#initialize" do
    context "when using just default initializer" do
      it "does not raise" do
        expect { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(name: "test") }.not_to raise_error
      end
    end
  end
end
