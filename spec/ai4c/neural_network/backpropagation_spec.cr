require "./../../spec_helper"
require "../../support/neural_network/data/training_patterns"
require "../../support/neural_network/data/patterns_with_noise"
require "../../support/neural_network/data/patterns_with_base_noise"

describe Ai4c::NeuralNetwork::Backpropagation do
  describe "#init_network" do
    describe "when given a net with structure of [4, 2]" do
      structure = [4, 2]
      expected_net = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0]]
      expected_weights_size = 1
      expected_weights_first_size = 5
      expected_weights_first_sub_size = 2
      net = Ai4c::NeuralNetwork::Backpropagation.new(structure).init_network

      it "sets @activation_nodes to expected nested array" do
        net.activation_nodes.should eq(expected_net)
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
    end

    describe "when given a net with structure of [2, 2, 1]" do
      structure = [2, 2, 1]
      expected_net = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0]]
      expected_weights_size = 2
      expected_weights_first_size = 3
      expected_weights_first_sub_size = 2
      net = Ai4c::NeuralNetwork::Backpropagation.new(structure).init_network

      it "sets @activation_nodes to expected nested array" do
        net.activation_nodes.should eq(expected_net)
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
    end

    describe "when given a net with structure of [2, 2, 1] with bias disabled" do
      structure = [2, 2, 1]
      expected_net = [[1.0, 1.0], [1.0, 1.0], [1.0]]
      expected_weights_size = 2
      expected_weights_first_size = 2 # one less than prev example since bias is disabled here
      expected_weights_first_sub_size = 2
      net = Ai4c::NeuralNetwork::Backpropagation.new(structure).init_network
      net.disable_bias = true
      net.init_network

      it "sets @activation_nodes to expected nested array" do
        net.activation_nodes.should eq(expected_net)
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
    end
  end

  describe "#eval" do
    describe "when given a net with structure of [3, 2]" do
      it "returns output nodes of expected size" do
        in_size = 3
        out_size = 2
        inputs = [3, 2, 3]
        structure = [in_size, out_size]
        net = Ai4c::NeuralNetwork::Backpropagation.new(structure)
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
        net = Ai4c::NeuralNetwork::Backpropagation.new(structure)
        y = net.eval(inputs)
        y.size.should eq(out_size)
      end
    end
  end

  describe "#dump" do
    describe "when given a net with structure of [3, 2]" do
      structure = [3, 2]
      net = Ai4c::NeuralNetwork::Backpropagation.new([3, 2]).init_network
      # s = Marshal.dump(net)
      # x = Marshal.load(s)
      # s = net.to_json
      # x = Ai4c::NeuralNetwork::Backpropagation.from_json(s)
      s = net.marshal_dump
      structure = s[:structure]
      x = Ai4c::NeuralNetwork::Backpropagation.new(structure).init_network
      x.marshal_load(s)

      it "@structure of the dumped net matches @structure of the loaded net" do
        assert_equality_of_nested_list net.structure, x.structure
      end

      it "@disable_bias on the dumped net matches @disable_bias of the loaded net" do
        net.disable_bias.should eq(x.disable_bias)
      end

      it "@learning_rate of the dumped net approximately matches @learning_rate of the loaded net" do
        assert_approximate_equality net.learning_rate, x.learning_rate
      end

      it "@momentum of the dumped net approximately matches @momentum of the loaded net" do
        assert_approximate_equality net.momentum, x.momentum
      end

      it "@weights of the dumped net approximately matches @weights of the loaded net" do
        assert_approximate_equality_of_nested_list net.weights, x.weights
      end

      it "@last_changes of the dumped net approximately matches @last_changes of the loaded net" do
        assert_approximate_equality_of_nested_list net.last_changes, x.last_changes
      end

      it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
        assert_approximate_equality_of_nested_list net.activation_nodes, x.activation_nodes
      end
    end
  end

  describe "#train" do
    describe "using image data (input) and shape flags (output) for triangle, square, and cross" do
      correct_count = 0

      error_averages = [] of Float64
      is_a_triangle = [1.0, 0.0, 0.0]
      is_a_square = [0.0, 1.0, 0.0]
      is_a_cross = [0.0, 0.0, 1.0]

      tr_input = TRIANGLE.flatten.map { |input| input.to_f / 5.0 }
      sq_input = SQUARE.flatten.map { |input| input.to_f / 5.0 }
      cr_input = CROSS.flatten.map { |input| input.to_f / 5.0 }

      tr_with_noise = TRIANGLE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
      sq_with_noise = SQUARE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
      cr_with_noise = CROSS_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }

      tr_with_base_noise = TRIANGLE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
      sq_with_base_noise = SQUARE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
      cr_with_base_noise = CROSS_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }

      net = Ai4c::NeuralNetwork::Backpropagation.new([256, 3])
      net.learning_rate = rand
      qty = 10 + (rand * 100).to_i

      describe "and training #{qty} times each at a learning rate of #{net.learning_rate.round(6)}" do
        qty.times do |i|
          errors = {} of Symbol => Float64
          [:tr, :sq, :cr].shuffle.each do |s|
            case s
            when :tr
              errors[:tr] = net.train(tr_input, is_a_triangle)
            when :sq
              errors[:sq] = net.train(sq_input, is_a_square)
            when :cr
              errors[:cr] = net.train(cr_input, is_a_cross)
            end
          end
          error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
        end

        describe "error_averages" do
          it "decrease (i.e.: first > last)" do
            (error_averages.first > error_averages.last).should eq(true)
          end

          it "should end up close to 0.1 +/- 0.1" do
            assert_approximate_equality(error_averages.last, 0.1, 0.1)
          end

          it "should end up close to 0.01 +/- 0.01" do
            assert_approximate_equality(error_averages.last, 0.01, 0.01)
          end

          it "should end up close to 0.001 +/- 0.001" do
            assert_approximate_equality(error_averages.last, 0.001, 0.001)
          end
        end

        describe "#eval correctly guesses shape flags (output) when given image data (input) of" do
          describe "original input data for" do
            it "TRIANGLE" do
              next_guess = guess(net, tr_input)
              check_guess(next_guess, "TRIANGLE")
            end

            it "SQUARE" do
              next_guess = guess(net, sq_input)
              check_guess(next_guess, "SQUARE")
            end

            it "CROSS" do
              next_guess = guess(net, cr_input)
              check_guess(next_guess, "CROSS")
            end
          end

          describe "noisy input data for" do
            it "TRIANGLE" do
              next_guess = guess(net, tr_with_noise)
              check_guess(next_guess, "TRIANGLE")
            end

            it "SQUARE" do
              next_guess = guess(net, sq_with_noise)
              check_guess(next_guess, "SQUARE")
            end

            it "CROSS" do
              next_guess = guess(net, cr_with_noise)
              check_guess(next_guess, "CROSS")
            end
          end

          describe "base noisy input data for" do
            it "TRIANGLE" do
              next_guess = guess(net, tr_with_base_noise)
              check_guess(next_guess, "TRIANGLE")
            end

            it "SQUARE" do
              next_guess = guess(net, sq_with_base_noise)
              check_guess(next_guess, "SQUARE")
            end

            it "CROSS" do
              next_guess = guess(net, cr_with_base_noise)
              check_guess(next_guess, "CROSS")
            end
          end
        end
      end
    end
  end
end
