require "./../../spec_helper"
require "../../support/neural_network/data/training_patterns"
require "../../support/neural_network/data/patterns_with_noise"
require "../../support/neural_network/data/patterns_with_base_noise"

describe Ai4cr::NeuralNetwork::Backpropagation2 do
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

      net = Ai4cr::NeuralNetwork::Backpropagation2.new([256, 3])
      net.learning_rate = rand
      qty = 100000 # + (rand * 100).to_i

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
