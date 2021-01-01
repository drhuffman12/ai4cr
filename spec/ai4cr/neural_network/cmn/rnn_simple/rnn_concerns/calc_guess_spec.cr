require "./../../../../../spec_helper"
require "./../../../../../spectator_helper"

# Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnSimple do
Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::CalcGuess do
  describe "#eval" do
    let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

    let(expected_outputs_guessed_before) { [[0.0], [0.0]] }
    let(expected_outputs_guessed_after) { [[0.1], [0.2]] }
    
    let(input_set_given) {
      [
        [0.1, 0.2],
        [0.3, 0.4]
      ]
    }
    # let(guess) { rnn_simple.eval(input_set_given) }

    context "before" do
      it "outputs_guessed start off all zero's" do
        expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_before)
      end
    end

    context "during" do
      it "calls 'step_load_inputs'"do
        allow(rnn_simple).to receive(:step_load_inputs).with(input_set_given).and_call_original
        expect(rnn_simple).to receive(:step_load_inputs).with(input_set_given)

        rnn_simple.eval(input_set_given)
      end

      it "calls 'step_calc_forward'"do
        expect(rnn_simple).to receive(:step_calc_forward)

        rnn_simple.eval(input_set_given)
      end
    end

    context "after" do
      it "returns expected outputs" do
        expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_before)

        # guess = 
        rnn_simple.eval(input_set_given)

        expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_after)
      end
    end
  end
end
