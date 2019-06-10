require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Backpropagation::Net do
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
      net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network

      it "sets @state to an Ai4cr::NeuralNetwork::Backpropagation::State" do
        net.state.nil?.should eq(false)
        net.state.class.should eq(Ai4cr::NeuralNetwork::Backpropagation::State)
      end

      describe "#train" do
        it "returns a Float64" do
          # puts "BEFORE net.to_json: #{net.to_json}\n training_stats: #{net.training_stats(in_bw: true)}"
          net.train(inputs, outputs).should be_a(Float64)
          # puts "AFTER net.to_json: #{net.to_json}\n training_stats: #{net.training_stats(in_bw: true)}"
        end

        it "updates the net" do
          net.train(inputs, outputs)
          net.state.activation_nodes.should_not eq(expected_activation_nodes_initialized)
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
      net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network

      # it "sets @state to an Ai4cr::NeuralNetwork::Backpropagation::State" do
      #   net.state.nil?.should eq(false)
      #   net.state.class.should eq(Ai4cr::NeuralNetwork::Backpropagation::State)
      # end

      describe "when exported as json" do
        net_exported = net.to_json
        net_exported_reimported = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(net_exported)

        describe "and re-imported from json" do
          net_exported_reimported = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(net_exported)
          net_exported_reimported_reexported = net_exported_reimported.to_json

          it "has a @state" do
            net.state.nil?.should eq(false)
            net.state.class.should eq(Ai4cr::NeuralNetwork::Backpropagation::State)
          end
          
          it "exported json matches reimported json" do
            JSON.parse(net_exported).should eq(JSON.parse(net_exported_reimported_reexported))
          end
        end
      end

      describe "#train" do
        it "returns a Float64" do
          net.train(inputs, outputs).should be_a(Float64)
        end

        it "updates the net" do
          net.train(inputs, outputs)
          net.state.activation_nodes.should_not eq(expected_activation_nodes_initialized)
        end
      end

      describe "when exported as json" do
        net_exported = net.to_json
        net_exported_reimported = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(net_exported)

        describe "and re-imported from json" do
          net_exported_reimported = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(net_exported)
          net_exported_reimported_reexported = net_exported_reimported.to_json

          it "has a @state" do
            net.state.nil?.should eq(false)
            net.state.class.should eq(Ai4cr::NeuralNetwork::Backpropagation::State)
          end
    
          it "exported json matches reimported json" do
            JSON.parse(net_exported).should eq(JSON.parse(net_exported_reimported_reexported))
          end
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
      net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network

      # puts "\nnet initialized: #{net.pretty_inspect}\n"

      expected_deltas_trained = [[0.0, 0.11817556435647361], [0.14770540994269726, -0.07628320427387962], [0.6131166367108101]]

      it "sets @state to an Ai4cr::NeuralNetwork::Backpropagation::State" do
        net.state.nil?.should eq(false)
        net.state.class.should eq(Ai4cr::NeuralNetwork::Backpropagation::State)
      end

      describe "#train" do
        it "returns a Float64" do
          net.train(inputs, outputs).should be_a(Float64)
          # puts "\nnet trained: #{net.pretty_inspect}\n"
        end

        it "updates the activation_nodes" do
          net.train(inputs, outputs)
          net.state.activation_nodes.should_not eq(expected_activation_nodes_initialized)
        end

        it "updates the the first set of activation_nodes (initially loaded with the inputs)" do
          net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network

          # initializers
          expected_activation_nodes_for_input = expected_activation_nodes_initialized.first[0..(inputs.size-1)].clone
          expected_activation_nodes_for_input.should_not eq(inputs)

          # actual_activation_nodes_for_input
          net.state.activation_nodes.first.should eq(expected_activation_nodes_initialized.first)
          net.state.activation_nodes.first[0..(inputs.size-1)].should_not eq(inputs)

          net.train(inputs, outputs)

          # after training
          net.state.activation_nodes.first.should_not eq(expected_activation_nodes_initialized.first)
          net.state.activation_nodes.first[0..(inputs.size-1)].should eq(inputs)
        end

        it "updates the the input_deltas" do
          net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network

          net.train(inputs, outputs)
          # after training
          input_deltas1 = net.state.input_deltas.clone
          net.state.input_deltas.size.should eq(inputs.size)

          net.train(inputs, outputs)
          # after training
          input_deltas2 = net.state.input_deltas.clone
          net.state.input_deltas.size.should eq(inputs.size)
          input_deltas1.should_not eq(input_deltas2)

          input_deltas1.size.should eq(input_deltas2.size)
          input_deltas1.should_not eq(input_deltas2)
        end

        it "updates the activation_nodes to different vales each training session" do
         activation_nodes_before = net.state.activation_nodes.clone
          net.train(inputs, outputs)
         activation_nodes_mid = net.state.activation_nodes.clone
          net.train(inputs, outputs)
         activation_nodes_after = net.state.activation_nodes.clone
         activation_nodes_mid.should_not eq(activation_nodes_before)
         activation_nodes_after.should_not eq(activation_nodes_mid)
         activation_nodes_after.should_not eq(activation_nodes_before)
        end

        it "updates the deltas to different vales each training session" do
         deltas_before = net.state.deltas.clone
          net.train(inputs, outputs)
         deltas_mid = net.state.deltas.clone
          net.train(inputs, outputs)
         deltas_after = net.state.deltas.clone
         deltas_mid.should_not eq(deltas_before)
         deltas_after.should_not eq(deltas_mid)
         deltas_after.should_not eq(deltas_before)
        end

        it "updates the @weights to different vales each training session" do
          weights_before = net.state.weights.clone
          net.train(inputs, outputs)
          weights_mid = net.state.weights.clone
          net.train(inputs, outputs)
          weights_after = net.state.weights.clone
          weights_mid.should_not eq(weights_before)
          weights_after.should_not eq(weights_mid)
          weights_after.should_not eq(weights_before)
        end
      end
    end
  end

  describe "#eval" do
    describe "when given a net with structure of [3, 2]" do
      it "returns output nodes of expected size" do
        in_size = 3
        out_size = 2
        inputs = [3, 2, 3]
        structure = [in_size, out_size]
        net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure)
        y = net.eval(inputs)
        y.size.should eq(out_size)
      end
    end

    describe "when given a net with structure of [2, 4, 8, 10, 7]" do
      it "returns output nodes of expected size" do
        in_size = 2
        layer_sizes = [4, 8, 10]
        out_size = 7
        structure = [in_size] + layer_sizes + [out_size]
        inputs = [2, 3]
        net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure)
        y = net.eval(inputs)
        y.size.should eq(out_size)
      end
    end
  end

  describe "#dump" do
    describe "when given a net with structure of [3, 2]" do
      structure = [3, 2]
      # net = Ai4cr::NeuralNetwork::Backpropagation::Net.new([3, 2]).init_network

      # TODO: Remove (marshal_dump and marshal_load are deprecated)
      net = Ai4cr::NeuralNetwork::Backpropagation::Net.new([3, 2]) # .init_network
      s = net.marshal_dump
      structure = s[:structure]
      x = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure) # .init_network
      x.marshal_load(s)

      # NOTE: *_json replaces marshal_dump and marshal_load
      json = net.to_json
      net2 = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(json)

      # NOTE: To make a net of similar config, use config_* instead of *_json
      config = net.to_config
      net3 = Ai4cr::NeuralNetwork::Backpropagation::Net.from_config(config)

      it "@structure of the dumped net matches @structure of the loaded net" do
        assert_equality_of_nested_list net.state.config.structure, x.state.config.structure # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_equality_of_nested_list net.state.config.structure, net2.state.config.structure
        assert_equality_of_nested_list net.state.config.structure, net3.state.config.structure
      end

      it "@disable_bias on the dumped net matches @disable_bias of the loaded net" do
        net.state.config.disable_bias.should eq(x.state.config.disable_bias) # TODO: Remove (marshal_dump and marshal_load are deprecated)
        net.state.config.disable_bias.should eq(net2.state.config.disable_bias)
        net.state.config.disable_bias.should eq(net3.state.config.disable_bias)
      end

      it "@learning_rate of the dumped net approximately matches @learning_rate of the loaded net" do
        assert_approximate_equality net.state.config.learning_rate, x.state.config.learning_rate # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality net.state.config.learning_rate, net2.state.config.learning_rate
        assert_approximate_equality net.state.config.learning_rate, net3.state.config.learning_rate
      end

      it "@momentum of the dumped net approximately matches @momentum of the loaded net" do
        assert_approximate_equality net.state.config.momentum, x.state.config.momentum # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality net.state.config.momentum, net2.state.config.momentum
        assert_approximate_equality net.state.config.momentum, net3.state.config.momentum
      end

      it "@weights of the dumped net approximately matches @weights of the loaded net" do
        assert_approximate_equality_of_nested_list net.state.weights, x.state.weights # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.state.weights, net2.state.weights
      end

      it "@last_changes of the dumped net approximately matches @last_changes of the loaded net" do
        assert_approximate_equality_of_nested_list net.state.last_changes, x.state.last_changes # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.state.last_changes, net2.state.last_changes
      end

      it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
        assert_approximate_equality_of_nested_list net.state.activation_nodes, x.state.activation_nodes # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.state.activation_nodes, net2.state.activation_nodes
      end

      it "@calculated_error_latest of the dumped net approximately matches @calculated_error_latest of the loaded net" do
        assert_approximate_equality_of_nested_list net.state.calculated_error_latest, x.state.calculated_error_latest # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.state.calculated_error_latest, net2.state.calculated_error_latest
      end
    end
  end

  describe "#train" do
    describe "when given a net with structure of [3, 2]" do
      structure = [3, 2]
      net = Ai4cr::NeuralNetwork::Backpropagation::Net.new([3, 2]) # .init_network

      it "returns an error of type Float64" do
        inputs = [1, 2, 3]
        outputs = [4, 5]
        error_value = net.train(inputs, outputs)
        error_value.should be_a(Float64)
      end
    end
  end
end
