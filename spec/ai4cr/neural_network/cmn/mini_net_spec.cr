require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::MiniNet do
  describe "when importing and exporting as JSON" do
    [
      Ai4cr::NeuralNetwork::LS_PRELU,
      Ai4cr::NeuralNetwork::LS_RELU,
      Ai4cr::NeuralNetwork::LS_SIGMOID,
      Ai4cr::NeuralNetwork::LS_TANH,
    ].each do |learning_styles|
      context "when given height: 2, width: 3, learning_styles: #{learning_styles}" do
        expected_keys = [
          "width", "height",
          "height_considering_bias",
          "width_indexes", "height_indexes",
          "inputs_given", "outputs_guessed",
          "weights", "last_changes",
          "error_stats",
          "outputs_expected",
          "input_deltas", "output_deltas",
          "bias_disabled", "learning_rate",
          "momentum",
          "learning_styles", "deriv_scale",
        ]

        context "when exporting to JSON" do
          np1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 2, width: 3, learning_styles: learning_styles)
          np1_json = np1.to_json
          np1_hash = JSON.parse(np1_json).as_h

          expected_keys.each do |key|
            it "it has top level key of #{key}" do
              (np1_hash.keys).should contain(key)
            end
          end
        end

        context "when importing from JSON" do
          np1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(2, 3, learning_styles)
          np1_json = np1.to_json

          np2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(np1_json)
          np2_json = np2.to_json

          # FYI: Due to some rounding errors during export/import, the following might not work:
          # it "re-exported JSON matches imported JSON" do
          #   (np1_json).should eq(np2_json)
          # end

          # However, it seems to be fine when you split it out by top-level keys:
          expected_keys.each do |key|
            it "re-exported JSON matches imported JSON for top level key of #{key}" do
              (np1_json[key]).should eq(np2_json[key])
            end
          end

          # And, it seems to be fine when you convert to hash:
          np1_hash = JSON.parse(np1_json).as_h
          np2_hash = JSON.parse(np2_json).as_h
          # FYI: Due to some rounding errors during export/import, the following might not work:
          pending "re-exported JSON matches imported JSON" do
            # NOTE: For now, mark as 'pending', but ...
            #   There are float rounding discrepancies between to/from json values.
            #   For this and other reasone, we should split up the below test to
            #   parse the JSON and validate individual values
            (np1_hash).should eq(np2_hash)
          end
        end
      end
    end
  end

  # NOTE Below are all for learing style Sigmoid; tests should be added to cover the other learning styles
  describe "#eval" do
    describe "when given a net with structure of [3, 2]" do
      net = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 3, width: 2, learning_styles: Ai4cr::NeuralNetwork::LS_SIGMOID)

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

        outputs_guessed_before = net.outputs_guessed.clone

        assert_equality_of_nested_list outputs_guessed_before, expected_outputs_guessed_before
      end

      it "the 'outputs_guessed' start are updated as expected" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone

        net.eval(inputs)
        outputs_guessed_after = net.outputs_guessed.clone

        assert_approximate_equality_of_nested_list outputs_guessed_after, expected_outputs_guessed_after
      end
    end
  end

  describe "#train" do
    describe "when given a net with structure of [3, 2]" do
      net = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 3, width: 2, learning_styles: Ai4cr::NeuralNetwork::LS_SIGMOID)
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

      it "output_deltas start as zeros" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone

        deltas_before = net.output_deltas.clone

        assert_equality_of_nested_list deltas_before, expected_deltas_before
      end

      it "correctly updates the output_deltas" do
        net.init_network
        net.learning_rate = 0.25
        net.momentum = 0.1
        net.weights = hard_coded_weights.clone

        deltas_before = net.output_deltas.clone
        net.train(inputs, outputs)
        deltas_after = net.output_deltas.clone

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

        net.train(inputs, outputs)

        weights_after = net.weights.clone

        weights_before.should eq(hard_coded_weights)

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
