require "./../../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::MiniNetConcerns::CalcGuess do
  # NOTE Below are all for learing style Sigmoid; tests should be added to cover the other learning styles
  describe "#eval" do
    describe "when given a net with structure of [3, 2]" do
      # before_each do
      # structure = [3, 2]
      # net = Ai4cr::NeuralNetwork::Backpropagation.new([3, 2])
      bias_scale = 1
      net = Ai4cr::NeuralNetwork::Cmn::MiniNet.new( height: 3, width: 2, bias_scale: bias_scale, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID )

      inputs = [0.1, 0.2, 0.3]
      hard_coded_weights = [
        [-0.9, 0.7],
        [-0.9, 0.6],
        [0.1, 0.2],
        [0.6, -0.3],
      ]
      expected_outputs_guessed_before = net.width.times.to_a.map { 0.0 }
      expected_outputs_guessed_after = [0.589040434058665, 0.48750260351579]

      it "the 'outputs_guessed' start as zeros" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone
        puts "\nnet (BEFORE): #{net.to_json}\n"

        outputs_guessed_before = net.outputs_guessed.clone

        assert_equality_of_nested_list outputs_guessed_before, expected_outputs_guessed_before
      end

      it "the 'outputs_guessed' start are updated as expected" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone
        puts "\nnet (BEFORE): #{net.to_json}\n"

        # outputs_guessed_before = net.outputs_guessed.clone

        net.eval(inputs)
        outputs_guessed_after = net.outputs_guessed.clone
        puts "\nnet (AFTER): #{net.to_json}\n"

        assert_approximate_equality_of_nested_list outputs_guessed_after, expected_outputs_guessed_after
      end
    end
  end
end
