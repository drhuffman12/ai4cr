require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::Chain do
  # describe "when given two nets with structure of [3, 4] and [4, 2]" do
  #   # before_each do
  #   # structure = [3, 2]
  #   # net = Ai4cr::NeuralNetwork::Backpropagation.new([3, 2])
  #   inputs = [0.1, 0.2, 0.3]

  #   hard_coded_weights0 = [
  #     [-0.4, 0.9, -0.4, -0.7],
  #     [0.1, 0.8, 0.9, -0.0],
  #     [-0.7, -0.3, -0.6, -0.7],
  #     [1.0, 0.2, 0.6, -0.5],
  #   ]
  #   hard_coded_weights1 = [
  #     [-0.4, 0.8],
  #     [-1.0, -0.3],
  #     [-0.6, 0.6],
  #     [0.2, -0.3]
  #     # ,
  #     # [1.0, -0.1],
  #   ]

  #   puts "hard_coded_weights0: #{hard_coded_weights0.each { |a| puts a.join("\t") }}"
  #   puts "hard_coded_weights1: #{hard_coded_weights1.each { |a| puts a.join("\t") }}"

  #   expected_outputs_guessed_before = [0.0, 0.0]
  #   expected_outputs_guessed_after = [0.454759979898907, 0.635915600435646]
  #   expected_outputs_guessed_trained = [1.0, 0.1]

  #   context "#init_network" do
  #     it "the 'outputs_guessed' start as zeros" do
  #       net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 3, width: 4, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  #       net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 4, width: 2, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  #       arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  #       arr << net0
  #       arr << net1
  #       cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  #       puts "net0.weights: #{net0.weights.map { |a| a.map { |b| b.round(1) } }}"
  #       puts "net1.weights: #{net1.weights.map { |a| a.map { |b| b.round(1) } }}"

  #       net0.init_network
  #       net0.learning_rate = 0.25
  #       net0.momentum = 0.1
  #       net0.weights = hard_coded_weights0.clone
  #       # puts "\nnet0 (BEFORE): #{net0.to_json}\n"

  #       net1.init_network
  #       net1.learning_rate = 0.25
  #       net1.momentum = 0.1
  #       net1.weights = hard_coded_weights1.clone
  #       # puts "\nnet1 (BEFORE): #{net1.to_json}\n"

  #       puts "\ncns (BEFORE): #{cns.to_json}\n"

  #       outputs_guessed_before = net1.outputs_guessed.clone

  #       assert_equality_of_nested_list outputs_guessed_before, expected_outputs_guessed_before
  #     end
  #   end

  #   context "#eval" do
  #     it "the 'outputs_guessed' are updated as expected" do
  #       net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 3, width: 4, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  #       net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 4, width: 2, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  #       arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  #       arr << net0
  #       arr << net1
  #       cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  #       puts "net0.weights: #{net0.weights.map { |a| a.map { |b| b.round(1) } }}"
  #       puts "net1.weights: #{net1.weights.map { |a| a.map { |b| b.round(1) } }}"

  #       net0.init_network
  #       net0.learning_rate = 0.25
  #       net0.momentum = 0.1
  #       net0.weights = hard_coded_weights0.clone
  #       # puts "\nnet0 (BEFORE): #{net0.to_json}\n"

  #       net1.init_network
  #       net1.learning_rate = 0.25
  #       net1.momentum = 0.1
  #       net1.weights = hard_coded_weights1.clone
  #       # puts "\nnet1 (BEFORE): #{net1.to_json}\n"

  #       puts "\ncns (BEFORE): #{cns.to_json}\n"

  #       # outputs_guessed_before = cns.net_set.last.outputs_guessed.clone

  #       cns.eval(inputs)
  #       outputs_guessed_after = cns.net_set.last.outputs_guessed.clone
  #       puts "\ncns (AFTER): #{cns.to_json}\n"

  #       assert_approximate_equality_of_nested_list expected_outputs_guessed_after, outputs_guessed_after
  #     end
  #   end

  #   context "#train" do
  #     it "the 'outputs_guessed' are updated as expected" do
  #       net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 3, width: 4, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  #       net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 4, width: 2, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID)
  #       arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  #       arr << net0
  #       arr << net1
  #       cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  #       puts "net0.weights: #{net0.weights.map { |a| a.map { |b| b.round(1) } }}"
  #       puts "net1.weights: #{net1.weights.map { |a| a.map { |b| b.round(1) } }}"

  #       net0.init_network
  #       net0.learning_rate = 0.25
  #       net0.momentum = 0.1
  #       net0.weights = hard_coded_weights0.clone
  #       # puts "\nnet0 (BEFORE): #{net0.to_json}\n"

  #       net1.init_network
  #       net1.learning_rate = 0.25
  #       net1.momentum = 0.1
  #       net1.weights = hard_coded_weights1.clone
  #       # puts "\nnet1 (BEFORE): #{net1.to_json}\n"

  #       puts "\ncns (BEFORE): #{cns.to_json}\n"

  #       # outputs_guessed_before = cns.net_set.last.outputs_guessed.clone

  #       cns.train(inputs, expected_outputs_guessed_trained)
  #       outputs_guessed_after = cns.net_set.last.outputs_guessed.clone
  #       puts "\ncns (AFTER): #{cns.to_json}\n"

  #       assert_approximate_equality_of_nested_list expected_outputs_guessed_after, outputs_guessed_after
  #     end
  #   end
  # end

  describe "when given two nets with structure of [2, 4] and [4, 3]" do
    # before_each do
    # structure = [3, 2]
    # net = Ai4cr::NeuralNetwork::Backpropagation.new([3, 2])
    inputs = [0.1, 0.2]

    hard_coded_weights0 = [
      [-0.4, 0.9, -0.4, -0.7], # index 0
      [0.1, 0.8, 0.9, -0.0], # index 1
      [-0.7, -0.3, -0.6, -0.7], # index bias
      # [1.0, 0.2, 0.6, -0.5],
    ]
    hard_coded_weights1 = [
      [-0.4, 0.8, 0.2],
      [-1.0, -0.3, 0.4],
      [-0.6, 0.6, 0.6],
      [0.2, -0.3, 0.8],
      # [1.0, -0.1],
    ]

    # puts "hard_coded_weights0: #{hard_coded_weights0.pretty_inspect}"
    # puts "hard_coded_weights1: #{hard_coded_weights1.pretty_inspect}"

    expected_outputs_guessed_before = [0.0, 0.0, 0.0]
    # expected_outputs_guessed_after = [0.454759979898907, 0.635915600435646]
    expected_outputs_guessed_after = [0.3127367832076713, 0.5628929575130488, 0.6782747272874269]
    expected_outputs_guessed_trained = [1.0, 0.1, 0.5]
      #  Expected 0.42809735352948664 to be within 0.01 of 0.635915600435646
      #  Expected 0.7010058764829652 to be within 0.01 of 0.454759979898907

    context "#init_network" do
      it "the 'outputs_guessed' start as zeros" do 
        # prep net vvv
        net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 2, width: 4, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID, disable_bias: false)
        net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 4, width: 3, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID, disable_bias: true)

        net0.init_network
        net0.learning_rate = 0.25
        net0.momentum = 0.1
        net0.weights = hard_coded_weights0.clone

        net1.init_network
        net1.learning_rate = 0.25
        net1.momentum = 0.1
        net1.weights = hard_coded_weights1.clone

        arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
        arr << net0
        arr << net1
        cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
        # prep net ^^^
     
        cns.validate.should be_true
        cns.weight_height_mismatches.should be_empty

        outputs_guessed_before = cns.net_set.last.outputs_guessed.clone

        assert_equality_of_nested_list expected_outputs_guessed_before, outputs_guessed_before
      end
    end

    context "#eval" do
      it "the 'outputs_guessed' are updated as expected" do
        # prep net vvv
        net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 2, width: 4, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID, disable_bias: false)
        net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 4, width: 3, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID, disable_bias: true)

        net0.init_network
        net0.learning_rate = 0.25
        net0.momentum = 0.1
        net0.weights = hard_coded_weights0.clone

        net1.init_network
        net1.learning_rate = 0.25
        net1.momentum = 0.1
        net1.weights = hard_coded_weights1.clone

        arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
        arr << net0
        arr << net1
        cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
        # prep net ^^^
     
        cns.validate.should be_true
        cns.weight_height_mismatches.should be_empty

        cns.eval(inputs)
        outputs_guessed_after = cns.net_set.last.outputs_guessed.clone
        # puts "\ncns (AFTER): #{cns.to_json}\n"

        assert_approximate_equality_of_nested_list expected_outputs_guessed_after, outputs_guessed_after
      end
    end

    context "#train" do
      it "the 'outputs_guessed' are updated as expected" do
        # prep net vvv
        net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 2, width: 4, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID, disable_bias: false)
        net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 4, width: 3, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID, disable_bias: true)

        net0.init_network
        net0.learning_rate = 0.25
        net0.momentum = 0.1
        net0.weights = hard_coded_weights0.clone

        net1.init_network
        net1.learning_rate = 0.25
        net1.momentum = 0.1
        net1.weights = hard_coded_weights1.clone

        arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
        arr << net0
        arr << net1
        cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
        # prep net ^^^
     
        cns.validate.should be_true
        cns.weight_height_mismatches.should be_empty

        cns.train(inputs, expected_outputs_guessed_trained)
        
        outputs_guessed_after = cns.net_set.last.outputs_guessed.clone
        # puts "\ncns (AFTER): #{cns.to_json}\n"

        assert_approximate_equality_of_nested_list expected_outputs_guessed_after, outputs_guessed_after

        assert_approximate_inequality_of_nested_list(hard_coded_weights0, net0.weights, delta = 0.000001)
      end
    end
  end

  # describe "when given a mix of Tanh, Prelu, Relu, and Sigmoid  MiniNets all chained together (with associated IO sizes)" do
  #   layer_0_size_without_bias = 3
  #   layer_1_size_without_bias = 4
  #   layer_2_size_without_bias = 5
  #   layer_3_size_without_bias = 6
  #   layer_4_size_without_bias = 7

  #   nt = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: layer_0_size_without_bias, width: layer_1_size_without_bias, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_TANH, disable_bias: false)
  #   nr = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: layer_1_size_without_bias, width: layer_2_size_without_bias, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_RELU, disable_bias: true)
  #   np = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: layer_2_size_without_bias, width: layer_3_size_without_bias, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_PRELU, disable_bias: true)
  #   ne = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: layer_3_size_without_bias, width: layer_4_size_without_bias, learning_style: Ai4cr::NeuralNetwork::Cmn::LS_SIGMOID, disable_bias: true)

  #   arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
  #   arr << nt
  #   arr << nr
  #   arr << np
  #   arr << ne
  #   cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

  #   initial_inputs = [rand, rand, rand]
  #   expected_inital_outputs = (arr.last.width.times.to_a.map { 0.0 })

  #   it "is valid" do
  #     puts "*"*8

  #     puts "cns: #{cns.pretty_inspect}"
  #     puts "*"*8

  #     puts "cns.validate!: #{cns.validate!}"
  #     puts "*"*8
  #     (cns.validate!).should be_true
  #     (cns.errors.empty?).should be_true
  #   end

  #   it "updates last net's outputs when guessing" do
  #     cns.net_set.each { |net| net.init_network }

  #     (cns.guesses_best).should eq(expected_inital_outputs)

  #     cns.eval(initial_inputs)

  #     (cns.guesses_best.size).should eq(expected_inital_outputs.size)
  #     (cns.guesses_best).should_not eq(expected_inital_outputs)
  #   end

  #   # it "exports to json without raising an error" do
  #   #   json_exported = cns.to_json
  #   #   # Below should not raise:
  #   #   cns = Ai4cr::NeuralNetwork::Cmn::Chain.from_json(json_exported)
  #   # end

  #   # it "imports exported json without raising an error" do
  #   #   json_exported = cns.to_json
  #   #   # Below should not raise:
  #   #   cns2 = Ai4cr::NeuralNetwork::Cmn::Chain.from_json(json_exported)
  #   #   # below should match
  #   #   (cns2.to_json).should eq(json_exported)
  #   # end

  # end
end
