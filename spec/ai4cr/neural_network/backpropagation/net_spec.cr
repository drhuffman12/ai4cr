require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Backpropagation do
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

      it "sets @activation_nodes to expected nested array" do
        net.activation_nodes.should eq(expected_activation_nodes_initialized)
      end

      it "sets @weights to expected size" do
        net.weights.size.should eq(expected_weights_size)
      end

      it "sets @weights.first to expected size" do
        net.weights.first.size.should eq(expected_weights_first_size)
      end

      it "sets each sub-array w/in @weights.first to expected size" do
        net.weights.first.each do |weights_n|
          weights_n.size.should eq(expected_weights_first_sub_size)
        end
      end

      describe "#train" do
        it "returns a Float64" do
          puts "BEFORE net.to_json: #{net.to_json}"
          net.train(inputs, outputs).should be_a(Float64)
          puts "AFTER net.to_json: #{net.to_json}"
        end

        it "updates the net" do
          net.train(inputs, outputs)
          net.activation_nodes.should_not eq(expected_activation_nodes_initialized)
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

      it "sets @activation_nodes to expected nested array" do
        net.activation_nodes.should eq(expected_activation_nodes_initialized)
      end

      it "sets @weights to expected size" do
        net.weights.size.should eq(expected_weights_size)
      end

      it "sets @weights.first to expected size" do
        net.weights.first.size.should eq(expected_weights_first_size)
      end

      it "sets each sub-array w/in @weights.first to expected size" do
        net.weights.first.each do |weights_n|
          weights_n.size.should eq(expected_weights_first_sub_size)
        end
      end

      describe "#train" do
        it "returns a Float64" do
          net.train(inputs, outputs).should be_a(Float64)
        end

        it "updates the net" do
          net.train(inputs, outputs)
          net.activation_nodes.should_not eq(expected_activation_nodes_initialized)
        end
      end
    end

    describe "when given a net with structure of [2, 2, 1] with bias disabled" do
      structure = [2, 2, 1]
      disable_bias = true
      inputs = [2, 3]
      outputs = [4]
      expected_activation_nodes_initialized = [[1.0, 1.0], [1.0, 1.0], [1.0]] # w/ disable_bias = true
      # expected_activation_nodes_initialized = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0]] # w/ disable_bias = false
      expected_activation_nodes_trained = [[0.0, 0.0], [0.0, 0.0], [0.0]]
      expected_weights_size = 2
      expected_weights_first_size = 2 # one less than prev example since bias is disabled here
      expected_weights_first_sub_size = 2
      expected_deltas_first_size = 0.0
      expected_deltas_initialized = [[0.0, 0.0], [0.0, 0.0], [0.0]]
      net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network
      # net.config.disable_bias = true
      # net.init_network

      puts "\nnet initialized: #{net.pretty_inspect}\n"

      # expected_weights_initialized_for_testing = [[[1.0,-1.0],[0.5, -0.5]],[[0.25],[-0.25]]]
      # net.weights = expected_weights_initialized_for_testing
      # allow(net).to receive(weights).and_return(expected_weights_initialized_for_testing)
      expected_deltas_trained = [[0.0, 0.11817556435647361], [0.14770540994269726, -0.07628320427387962], [0.6131166367108101]]

      it "sets @activation_nodes to expected nested array" do
        net.activation_nodes.should eq(expected_activation_nodes_initialized)
      end

      it "sets @weights to expected size" do
        net.weights.size.should eq(expected_weights_size)
      end

      it "sets @weights.first to expected size" do
        net.weights.first.size.should eq(expected_weights_first_size)
      end

      it "sets each sub-array w/in @weights.first to expected size" do
        net.weights.first.each do |weights_n|
          weights_n.size.should eq(expected_weights_first_sub_size)
        end
      end

      # it "sets @weights to expected DEBUG" do
      #   net.weights.should eq(expected_weights_initialized_for_testing)
      # end

      it "sets @deltas to expected size" do
        net.deltas.size.should eq(expected_deltas_initialized.size)
        net.deltas.first.size.should eq(expected_deltas_initialized.first.size)
      end

      it "sets @deltas to expected nested array" do
        net.deltas.should eq(expected_deltas_initialized)
      end

      describe "#train" do
        it "returns a Float64" do
          net.train(inputs, outputs).should be_a(Float64)
          puts "\nnet trained: #{net.pretty_inspect}\n"
        end

        it "updates the activation_nodes" do
          net.train(inputs, outputs)
          net.activation_nodes.should_not eq(expected_activation_nodes_initialized)
        end

        it "updates the the first set of activation_nodes (initially loaded with the inputs)" do
          net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network

          # initializers
          expected_activation_nodes_for_input = expected_activation_nodes_initialized.first[0..(inputs.size-1)].clone
          expected_activation_nodes_for_input.should_not eq(inputs)

          # actual_activation_nodes_for_input
          net.activation_nodes.first.should eq(expected_activation_nodes_initialized.first)
          net.activation_nodes.first[0..(inputs.size-1)].should_not eq(inputs)

          net.train(inputs, outputs)

          # after training
          net.activation_nodes.first.should_not eq(expected_activation_nodes_initialized.first)
          net.activation_nodes.first[0..(inputs.size-1)].should eq(inputs)
        end

        it "updates the the input_deltas" do
          net = Ai4cr::NeuralNetwork::Backpropagation::Net.new(structure, disable_bias) # .init_network

          net.train(inputs, outputs)
          # after training
          input_deltas1 = net.input_deltas.clone
          net.input_deltas.size.should eq(inputs.size)

          net.train(inputs, outputs)
          # after training
          input_deltas2 = net.input_deltas.clone
          net.input_deltas.size.should eq(inputs.size)
          input_deltas1.should_not eq(input_deltas2)

          input_deltas1.size.should eq(input_deltas2.size)
          input_deltas1.should_not eq(input_deltas2)
        end

        it "updates the activation_nodes to different vales each training session" do
         activation_nodes_before = net.activation_nodes.clone
          net.train(inputs, outputs)
         activation_nodes_mid = net.activation_nodes.clone
          net.train(inputs, outputs)
         activation_nodes_after = net.activation_nodes.clone
         activation_nodes_mid.should_not eq(activation_nodes_before)
         activation_nodes_after.should_not eq(activation_nodes_mid)
         activation_nodes_after.should_not eq(activation_nodes_before)
        end

        it "updates the deltas to different vales each training session" do
         deltas_before = net.deltas.clone
          net.train(inputs, outputs)
         deltas_mid = net.deltas.clone
          net.train(inputs, outputs)
         deltas_after = net.deltas.clone
         deltas_mid.should_not eq(deltas_before)
         deltas_after.should_not eq(deltas_mid)
         deltas_after.should_not eq(deltas_before)
        end

        it "updates the @weights to different vales each training session" do
          weights_before = net.weights.clone
          net.train(inputs, outputs)
          weights_mid = net.weights.clone
          net.train(inputs, outputs)
          weights_after = net.weights.clone
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
        assert_equality_of_nested_list net.config.structure, x.config.structure # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_equality_of_nested_list net.config.structure, net2.config.structure
        assert_equality_of_nested_list net.config.structure, net3.config.structure
      end

      it "@disable_bias on the dumped net matches @disable_bias of the loaded net" do
        net.config.disable_bias.should eq(x.config.disable_bias) # TODO: Remove (marshal_dump and marshal_load are deprecated)
        net.config.disable_bias.should eq(net2.config.disable_bias)
        net.config.disable_bias.should eq(net3.config.disable_bias)
      end

      it "@learning_rate of the dumped net approximately matches @learning_rate of the loaded net" do
        assert_approximate_equality net.config.learning_rate, x.config.learning_rate # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality net.config.learning_rate, net2.config.learning_rate
        assert_approximate_equality net.config.learning_rate, net3.config.learning_rate
      end

      it "@momentum of the dumped net approximately matches @momentum of the loaded net" do
        assert_approximate_equality net.config.momentum, x.config.momentum # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality net.config.momentum, net2.config.momentum
        assert_approximate_equality net.config.momentum, net3.config.momentum
      end

      it "@weights of the dumped net approximately matches @weights of the loaded net" do
        assert_approximate_equality_of_nested_list net.weights, x.weights # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.weights, net2.weights
      end

      it "@last_changes of the dumped net approximately matches @last_changes of the loaded net" do
        assert_approximate_equality_of_nested_list net.last_changes, x.last_changes # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.last_changes, net2.last_changes
      end

      it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
        assert_approximate_equality_of_nested_list net.activation_nodes, x.activation_nodes # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes
      end

      it "@calculated_error_total of the dumped net approximately matches @calculated_error_total of the loaded net" do
        assert_approximate_equality_of_nested_list net.calculated_error_total, x.calculated_error_total # TODO: Remove (marshal_dump and marshal_load are deprecated)
        assert_approximate_equality_of_nested_list net.calculated_error_total, net2.calculated_error_total
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
