require "./../../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::MiniNetConcerns::TrainAndAdjust do
 
  describe "#train" do
    describe "when given a net with structure of [3, 2]" do
      # before_each do
      # structure = [3, 2]
      # net = Ai4cr::NeuralNetwork::Backpropagation.new([3, 2])
      bias_scale = 1.0
      net = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(
        height: 3, width: 2,
        bias_scale: bias_scale,
        learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID
      )
      hard_coded_weights = [
        [-0.9, 0.7],
        [-0.9, 0.6],
        [0.1, 0.2],
        [0.6, -0.3],
      ]
      expected_deltas_before = [0.0, 0.0]
      expected_after_deltas = [-0.045761358353806764, 0.0031223972161964547]
      expected_after_weights = [
        [-0.9011440339588452, 0.7000780599304048],
        [-0.9022880679176903, 0.6001561198608099],
        [0.0965678981234645, 0.20023417979121474],
        [0.5885596604115483, -0.2992194006959509],
      ]
      expected_error = 0.017946235313986033

      inputs = [0.1, 0.2, 0.3]
      outputs = [0.4, 0.5]
      # end

      it "output_deltas start as zeros" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone
        puts "\nnet (BEFORE): #{net.to_json}\n"

        deltas_before = net.output_deltas.clone

        assert_equality_of_nested_list deltas_before, expected_deltas_before
      end

      it "correctly updates the output_deltas" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone
        puts "\nnet (BEFORE): #{net.to_json}\n"

        deltas_before = net.output_deltas.clone
        net.train(inputs, outputs)
        deltas_after = net.output_deltas.clone
        puts "\nnet (AFTER): #{net.to_json}\n"

        assert_equality_of_nested_list deltas_before, expected_deltas_before
        assert_approximate_equality_of_nested_list deltas_after, expected_after_deltas, 0.0000001
      end

      it "weights do change" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone

        weights_before = net.weights.clone
        net.train(inputs, outputs)
        weights_after = net.weights.clone
        assert_inequality_of_nested_list weights_before, weights_after
      end

      it "correctly updates the weights" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone

        weights_before = net.weights.clone
        puts "\nnet (BEFORE): #{net.to_json}\n"

        net.train(inputs, outputs)

        weights_after = net.weights.clone
        puts "\nnet (AFTER): #{net.to_json}\n"

        weights_before.should eq(hard_coded_weights)

        # weights_after.should eq(expected_after_weights)
        # assert_approximate_equality_of_nested_list weights_after, expected_after_weights, 0.0000001
        assert_approximate_equality_of_nested_list weights_after, expected_after_weights, 0.01
      end

      it "returns an error of type Float64" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone

        error_value = net.train(inputs, outputs)
        error_value.should be_a(Float64)
      end

      it "returns the expected error" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone

        error_value = net.train(inputs, outputs)
        error_value.should eq(expected_error)
      end
    end
  end
end
