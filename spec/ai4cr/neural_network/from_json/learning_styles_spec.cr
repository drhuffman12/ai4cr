require "./../../../spec_helper"
require "./../../../spectator_helper"

Spectator.describe "from_json" do
  describe "Ai4cr::NeuralNetwork::LearningStyle" do
    context "LS_PRELU" do
      let(ls_prelu) { Ai4cr::NeuralNetwork::LS_PRELU }
      let(ls) { ls_prelu }

      it "exports and imports" do
        fj = ls.to_json
        tj = Ai4cr::NeuralNetwork::LearningStyle.from_json(fj)

        expect(tj.to_json).to eq(ls.to_json)
      end
    end

    context "LS_PRELU" do
      let(ls_prelu) { Ai4cr::NeuralNetwork::LS_PRELU }
      let(ls_set) { [ls_prelu] }

      it "exports and imports" do
        fj = ls_set.to_json
        tj = Array(Ai4cr::NeuralNetwork::LearningStyle).from_json(fj)

        expect(tj.to_json).to eq(ls_set.to_json)
      end
    end

    context "LS_PRELU" do
      let(ls_prelu) { Ai4cr::NeuralNetwork::LS_PRELU }
      let(ls_relu) { Ai4cr::NeuralNetwork::LS_RELU }
      let(ls_set) { [ls_prelu, ls_relu] }

      it "exports and imports" do
        fj = ls_set.to_json
        tj = Array(Ai4cr::NeuralNetwork::LearningStyle).from_json(fj)

        expect(tj.to_json).to eq(ls_set.to_json)
      end
    end
  end
end
