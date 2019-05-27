require "json"
require "./../../../spec_helper"
require "../../../support/neural_network/data/training_patterns"
require "../../../support/neural_network/data/patterns_with_noise"
require "../../../support/neural_network/data/patterns_with_base_noise"

charter = AsciiBarCharter.new(0.0,1.0,3)

describe Ai4cr::NeuralNetwork::Backpropagation::Net do
  describe "#train" do
    describe "using image data (input) and shape flags (output) for triangle, square, and cross" do
      correct_count = 0

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

      qty = 400

      describe "with bias disabled" do
        input_size = 256
        disable_bias = true
        net = Ai4cr::NeuralNetwork::Backpropagation::Net.new([input_size, 3], learning_rate: rand, disable_bias: disable_bias)

        stats = net.training_stats(in_bw: true)
        puts "BEFORE any .. training_stats: #{stats}"

        # File.write("../ai4cr_ui/db/seeds/BackpropagationNet.new.json",net.to_json)

        it "does not include a bias node" do
          net.state.activation_nodes.first.size.should eq(input_size)
        end

        describe "and training #{qty} times each at a learning rate of #{net.state.config.learning_rate.round(6)}" do
          error_averages = [] of Float64
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
          stats = net.training_stats(in_bw: true)
          puts "AFTER some .. training_stats: #{stats}"
          
          File.write("../ai4cr_ui/db/seeds/BackpropagationNet.trained.json",net.to_json)
          
          describe "JSON (de-)serialization works" do
            it "@calculated_error_latest of the dumped net approximately matches @calculated_error_latest of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(json)

              assert_approximate_equality_of_nested_list net.state.calculated_error_latest, net2.state.calculated_error_latest, 0.000000001
            end
          
            it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(json)

              assert_approximate_equality_of_nested_list net.state.activation_nodes, net2.state.activation_nodes, 0.000000001
            end
          
            it "@weights of the dumped net approximately matches @weights of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation::Net.from_json(json)

              assert_approximate_equality_of_nested_list net.state.weights, net2.state.weights, 0.000000001
            end
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

          stats = net.training_stats(in_bw: true)
          puts "AFTER all .. training_stats: #{stats}"
        end
      end

      describe "with bias enabled" do
        input_size = 256
        disable_bias = false
        net = Ai4cr::NeuralNetwork::Backpropagation::Net.new([input_size, 3], learning_rate: rand, disable_bias: disable_bias)

        it "does not include a bias node" do
          net.state.activation_nodes.first.size.should eq(input_size + 1)
        end

        describe "and training #{qty} times each at a learning rate of #{net.state.config.learning_rate.round(6)}" do
          error_averages = [] of Float64
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

          puts "\n** error_averages when bias #{disable_bias ? "IS" : "is NOT"} disable and learning rate of #{net.state.config.learning_rate.round(2)}:\n  #{charter.bar_prefixed_with_number(error_averages.first)} .. #{charter.bar_prefixed_with_number(error_averages.last)} #{charter.plot(error_averages)}\n"

          puts "net.state.activation_nodes sizes: #{net.state.activation_nodes.map{|layer| layer.size}}"
          puts "net.state.deltas sizes: #{net.state.deltas.map{|layer| layer.size}}"
        end

        stats = net.training_stats(in_bw: true)
        puts "AFTER with bias enabled .. training_stats: #{stats}"
      end

      describe "with chained networks" do
        net1_input_size = 256
        net1_output_size = 127
        net2_output_size = 3

        net1_disable_bias = false # bias at beginning (if desired); not needed after
        net2_disable_bias = true

        net1 = Ai4cr::NeuralNetwork::Backpropagation::Net.new([net1_input_size, net1_output_size], learning_rate: rand, disable_bias: net1_disable_bias)

        net2 = Ai4cr::NeuralNetwork::Backpropagation::Net.new([net1_output_size, net2_output_size], learning_rate: rand, disable_bias: net2_disable_bias)

        describe "and training #{qty} times each at a learning rate of (net1) #{net1.state.config.learning_rate.round(3)} (net2) #{net2.state.config.learning_rate.round(3)}" do
          error_averages = [] of Float64
          qty.times do |i|
            errors = {} of Symbol => Float64
            [:tr, :sq, :cr].shuffle.each do |s|
              case s
              when :tr
                errors[:tr] = train_chained(net1, net2, tr_input, is_a_triangle)
              when :sq
                errors[:sq] = train_chained(net1, net2, sq_input, is_a_square)
              when :cr
                errors[:cr] = train_chained(net1, net2, cr_input, is_a_cross)
              end
            end
            error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
          end

          it "net1's output is the same size as net1's input_deltas" do
            net1.state.activation_nodes.last.size.should_not eq(net1.state.input_deltas.size)
          end

          it "net1's output is the same size as net2's input" do
            net1.state.activation_nodes.last.size.should eq(net2.state.config.structure.first)
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
                next_guess = guess_chained(net1, net2, tr_input)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = guess_chained(net1, net2, sq_input)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = guess_chained(net1, net2, cr_input)
                check_guess(next_guess, "CROSS")
              end
            end

            describe "noisy input data for" do
              it "TRIANGLE" do
                next_guess = guess_chained(net1, net2, tr_with_noise)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = guess_chained(net1, net2, sq_with_noise)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = guess_chained(net1, net2, cr_with_noise)
                check_guess(next_guess, "CROSS")
              end
            end

            describe "base noisy input data for" do
              it "TRIANGLE" do
                next_guess = guess_chained(net1, net2, tr_with_base_noise)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = guess_chained(net1, net2, sq_with_base_noise)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = guess_chained(net1, net2, cr_with_base_noise)
                check_guess(next_guess, "CROSS")
              end
            end
          end

          puts "\n** error_averages when chained with first having bias and 2nd not and learning rate of (net1) #{net1.state.config.learning_rate.round(2)} (net2) #{net2.state.config.learning_rate.round(2)}:\n  #{charter.bar_prefixed_with_number(error_averages.first)} .. #{charter.bar_prefixed_with_number(error_averages.last)} #{charter.plot(error_averages)}\n"
        end
      end
    end
  end
end
