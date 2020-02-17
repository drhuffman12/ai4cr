require "./../../../../spec_helper"

describe Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Sequencial do
  describe "when given two nets with structure of [3, 4] and [4, 2]" do
    # before_each do
    # structure = [3, 2]
    # net = Ai4cr::NeuralNetwork::Backpropagation.new([3, 2])
    inputs = [0.1, 0.2, 0.3]

    hard_coded_weights0 = [
      [-0.4, 0.9, -0.4, -0.7],
      [0.1, 0.8, 0.9, -0.0],
      [-0.7, -0.3, -0.6, -0.7],
      [1.0, 0.2, 0.6, -0.5],
    ]
    hard_coded_weights1 = [
      [-0.4, 0.8],
      [-1.0, -0.3],
      [-0.6, 0.6],
      [0.2, -0.3],
      [1.0, -0.1],
    ]

    puts "hard_coded_weights0: #{hard_coded_weights0.each { |a| puts a.join("\t") }}"
    puts "hard_coded_weights1: #{hard_coded_weights1.each { |a| puts a.join("\t") }}"

    expected_outputs_guessed_before = [0.0, 0.0]
    expected_outputs_guessed_after = [0.454759979898907, 0.635915600435646]

    it "the 'outputs_guessed' start as zeros" do
      net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(height: 3, width: 4)
      net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(height: 4, width: 2)
      cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Sequencial(Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp).new([net0, net1])

      puts "net0.weights: #{net0.weights.map { |a| a.map { |b| b.round(1) } }}"
      puts "net1.weights: #{net1.weights.map { |a| a.map { |b| b.round(1) } }}"

      net0.init_network
      net0.learning_rate = 0.25
      net0.momentum = 0.1
      net0.weights = hard_coded_weights0.clone
      # puts "\nnet0 (BEFORE): #{net0.to_json}\n"

      net1.init_network
      net1.learning_rate = 0.25
      net1.momentum = 0.1
      net1.weights = hard_coded_weights1.clone
      # puts "\nnet1 (BEFORE): #{net1.to_json}\n"

      puts "\ncns (BEFORE): #{cns.to_json}\n"

      outputs_guessed_before = net1.outputs_guessed.clone

      assert_equality_of_nested_list outputs_guessed_before, expected_outputs_guessed_before
    end

    it "the 'outputs_guessed' start are updated as expected" do
      net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(height: 3, width: 4)
      net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(height: 4, width: 2)
      cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Sequencial(Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp).new([net0, net1])

      puts "net0.weights: #{net0.weights.map { |a| a.map { |b| b.round(1) } }}"
      puts "net1.weights: #{net1.weights.map { |a| a.map { |b| b.round(1) } }}"

      net0.init_network
      net0.learning_rate = 0.25
      net0.momentum = 0.1
      net0.weights = hard_coded_weights0.clone
      # puts "\nnet0 (BEFORE): #{net0.to_json}\n"

      net1.init_network
      net1.learning_rate = 0.25
      net1.momentum = 0.1
      net1.weights = hard_coded_weights1.clone
      # puts "\nnet1 (BEFORE): #{net1.to_json}\n"

      puts "\ncns (BEFORE): #{cns.to_json}\n"

      outputs_guessed_before = cns.net_set.last.outputs_guessed.clone

      cns.eval(inputs)
      outputs_guessed_after = cns.net_set.last.outputs_guessed.clone
      puts "\ncns (AFTER): #{cns.to_json}\n"

      assert_approximate_equality_of_nested_list outputs_guessed_after, expected_outputs_guessed_after
    end
  end
end
