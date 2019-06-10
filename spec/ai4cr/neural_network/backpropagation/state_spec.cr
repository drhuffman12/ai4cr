require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Backpropagation::State do
  describe "#initialize" do
    describe "when given a net with structure of [4, 2] and bias enabled" do
      structure = [4, 2]
      disable_bias = false
      inputs = [1, 2, 3, 4]
      outputs = [5, 6]
      expected_activation_nodes_initialized = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0]]
      expected_weights_size = 1
      expected_weights_first_size = 5
      expected_weights_first_sub_size = 2
      state = Ai4cr::NeuralNetwork::Backpropagation::State.new(structure, disable_bias) # .init_network

      it "sets @activation_nodes to expected nested array" do
        state.activation_nodes.should eq(expected_activation_nodes_initialized)
      end

      it "sets @weights to expected size" do
        state.weights.size.should eq(expected_weights_size)
      end

      it "sets @weights.first to expected size" do
        state.weights.first.size.should eq(expected_weights_first_size)
      end

      it "sets each sub-array w/in @weights.first to expected size" do
        state.weights.first.each do |weights_n|
          weights_n.size.should eq(expected_weights_first_sub_size)
        end
      end
    end

    describe "when given a net with structure of [2, 2, 1] and bias enabled" do
      structure = [2, 2, 1]
      disable_bias = false
      inputs = [1, 2]
      outputs = [3]
      expected_activation_nodes_initialized = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0]]
      expected_weights_size = 2
      expected_weights_first_size = 3
      expected_weights_first_sub_size = 2
      state = Ai4cr::NeuralNetwork::Backpropagation::State.new(structure, disable_bias) # .init_network

      it "sets @activation_nodes to expected nested array" do
        state.activation_nodes.should eq(expected_activation_nodes_initialized)
      end

      it "sets @weights to expected size" do
        state.weights.size.should eq(expected_weights_size)
      end

      it "sets @weights.first to expected size" do
        state.weights.first.size.should eq(expected_weights_first_size)
      end

      it "sets each sub-array w/in @weights.first to expected size" do
        state.weights.first.each do |weights_n|
          weights_n.size.should eq(expected_weights_first_sub_size)
        end
      end

      describe "when exported as json" do
        state_exported = state.to_json
        # state_exported_reimported = Ai4cr::NeuralNetwork::Backpropagation::State.from_json(state_exported)

        describe "and re-imported from json" do
          state_exported_reimported = Ai4cr::NeuralNetwork::Backpropagation::State.from_json(state_exported)
          # state_exported_reimported_reexported = state_exported_reimported.to_json

          # expected_state = state # Ai4cr::NeuralNetwork::Backpropagation::State.from_json(state.to_json)

          describe "sets expected values for" do
            it "config" do
              # state.config.should eq(state.config)
              state_exported_reimported.config.should eq(state.config)
            end

            it "calculated_error_latest" do
              # assert_approximate_equality(state.calculated_error_latest, state.calculated_error_latest)
              assert_approximate_equality(state_exported_reimported.calculated_error_latest, state.calculated_error_latest)
            end

            it "track_history" do
              # state.track_history.should eq(state.track_history)
              state_exported_reimported.track_history.should eq(state.track_history)
            end

            it "weights" do
              # assert_approximate_equality_of_nested_list(state.weights, state.weights)
              assert_approximate_equality_of_nested_list(state_exported_reimported.weights, state.weights)
            end

            it "last_changes" do
              # assert_approximate_equality_of_nested_list(state.last_changes, state.last_changes)
              assert_approximate_equality_of_nested_list(state_exported_reimported.last_changes, state.last_changes)
            end

            it "activation_nodes" do
              # assert_approximate_equality_of_nested_list(state.activation_nodes, state.activation_nodes)
              assert_approximate_equality_of_nested_list(state_exported_reimported.activation_nodes, state.activation_nodes)
            end

            it "input_deltas" do
              # assert_approximate_equality_of_nested_list(state.input_deltas, state.input_deltas)
              assert_approximate_equality_of_nested_list(state_exported_reimported.input_deltas, state.input_deltas)
            end

            it "calculated_error_history" do
              # assert_approximate_equality_of_nested_list(state.calculated_error_history, state.calculated_error_history)
              assert_approximate_equality_of_nested_list(state_exported_reimported.calculated_error_history, state.calculated_error_history)
            end
          end

          # it "exported json matches reimported json" do
          #   state_exported_reparsed.should eq(state_reimported_reparsed)
          # end
          
          # it "sets @activation_nodes to expected nested array" do
          #   state_reimported.activation_nodes.should eq(expected_activation_nodes_initialized)
          # end

          # it "sets @weights to expected size" do
          #   state_reimported.weights.size.should eq(expected_weights_size)
          # end

          # it "sets @weights.first to expected size" do
          #   state_reimported.weights.first.size.should eq(expected_weights_first_size)
          # end

          # it "sets each sub-array w/in @weights.first to expected size" do
          #   state_reimported.weights.first.each do |weights_n|
          #     weights_n.size.should eq(expected_weights_first_sub_size)
          #   end
          # end
        end
      end
    end

    describe "when given a net with structure of [2, 2, 1] with bias disabled" do
      structure = [2, 2, 1]
      disable_bias = true
      inputs = [2, 3]
      outputs = [4]
      expected_activation_nodes_initialized = [[1.0, 1.0], [1.0, 1.0], [1.0]] # w/ disable_bias = true
      expected_activation_nodes_trained = [[0.0, 0.0], [0.0, 0.0], [0.0]]
      expected_weights_size = 2
      expected_weights_first_size = 2 # one less than prev example since bias is disabled here
      expected_weights_first_sub_size = 2
      expected_deltas_first_size = 0.0
      expected_deltas_initialized = [[0.0, 0.0], [0.0, 0.0], [0.0]]
      state = Ai4cr::NeuralNetwork::Backpropagation::State.new(structure, disable_bias) # .init_network

      # puts "\nnet initialized: #{net.pretty_inspect}\n"

      expected_deltas_trained = [[0.0, 0.11817556435647361], [0.14770540994269726, -0.07628320427387962], [0.6131166367108101]]

      it "sets @activation_nodes to expected nested array" do
        state.activation_nodes.should eq(expected_activation_nodes_initialized)
      end

      it "sets @weights to expected size" do
        state.weights.size.should eq(expected_weights_size)
      end

      it "sets @weights.first to expected size" do
        state.weights.first.size.should eq(expected_weights_first_size)
      end

      it "sets each sub-array w/in @weights.first to expected size" do
        state.weights.first.each do |weights_n|
          weights_n.size.should eq(expected_weights_first_sub_size)
        end
      end

      # it "sets @weights to expected DEBUG" do
      #   state.weights.should eq(expected_weights_initialized_for_testing)
      # end

      it "sets @deltas to expected size" do
        state.deltas.size.should eq(expected_deltas_initialized.size)
        state.deltas.first.size.should eq(expected_deltas_initialized.first.size)
      end

      it "sets @deltas to expected nested array" do
        state.deltas.should eq(expected_deltas_initialized)
      end
    end
  end

end
