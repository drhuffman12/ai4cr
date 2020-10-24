require "./../../spec_helper"
require "../../support/neural_network/data/*"
require "json"
require "ascii_bar_charter"

describe Ai4cr::NeuralNetwork::Backpropagation do
  describe "#train" do
    describe "with a shape of [256,3]" do
      describe "using image data (input) and shape flags (output) for triangle, square, and cross" do
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

        net = Ai4cr::NeuralNetwork::Backpropagation.new([256, 3], error_distance_history_max: 60)

        # net.learning_rate = rand
        qty = 20000
        qty_x_percent = qty // 5

        describe "and training #{qty} times each at a learning rate of #{net.learning_rate.round(6)}" do
          qty.times do |i|
            print "." if i % qty_x_percent == 0 # 1000 == 0
            errors = {} of Symbol => Float64
            [:tr, :sq, :cr].shuffle.each do |s|
              case s
              when :tr
                errors[:tr] = net.train(tr_input, is_a_triangle)
                net.step_calculate_error_distance_history if i % qty_x_percent == 0
              when :sq
                errors[:sq] = net.train(sq_input, is_a_square)
                net.step_calculate_error_distance_history if i % qty_x_percent == 0
              when :cr
                errors[:cr] = net.train(cr_input, is_a_cross)
                net.step_calculate_error_distance_history if i % qty_x_percent == 0
              end
            end
            error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
          end

          puts "\n--------\n"
          min = 0.0
          max = 1.0
          precision = 2.to_i8
          in_bw = false
          prefixed = false
          reversed = false

          charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
          plot = charter.plot(net.error_distance_history, prefixed)

          puts "#{net.class.name} with structure of #{net.structure}:"
          puts "  plot: '#{plot}'"
          puts "  error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"

          puts "\n--------\n"

          describe "JSON (de-)serialization works" do
            it "@calculated_error_total of the dumped net approximately matches @calculated_error_total of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation.from_json(json)

              assert_approximate_equality_of_nested_list net.calculated_error_total, net2.calculated_error_total, 0.000000001
            end

            it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation.from_json(json)

              assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
            end

            it "@weights of the dumped net approximately matches @weights of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation.from_json(json)

              assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
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
        end
      end
    end

    hidden_size = 500
    describe "with a shape of [256,#{hidden_size},#{hidden_size},3]" do
      describe "using image data (input) and shape flags (output) for triangle, square, and cross" do
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

        net = Ai4cr::NeuralNetwork::Backpropagation.new([256, hidden_size, hidden_size, 3], error_distance_history_max: 60)

        # net.learning_rate = rand
        qty = 20000
        qty_x_percent = qty // 5

        puts "\n--------\n"
        puts "#{net.class.name} with structure of #{net.structure}:"

        describe "and training #{qty} times each at a learning rate of #{net.learning_rate.round(6)}" do
          puts "\nTRAINING:\n"
          timestamp_before = Time.utc
          qty.times do |i|
            print "." if i % qty_x_percent == 0 # 1000 == 0
            errors = {} of Symbol => Float64
            [:tr, :sq, :cr].shuffle.each do |s|
              case s
              when :tr
                errors[:tr] = net.train(tr_input, is_a_triangle)
                net.step_calculate_error_distance_history if i % qty_x_percent == 0
              when :sq
                errors[:sq] = net.train(sq_input, is_a_square)
                net.step_calculate_error_distance_history if i % qty_x_percent == 0
              when :cr
                errors[:cr] = net.train(cr_input, is_a_cross)
                net.step_calculate_error_distance_history if i % qty_x_percent == 0
              end
            end
            error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
          end
          timestamp_after = Time.utc

          puts "\n--------\n"
          puts "duration: #{timestamp_after - timestamp_before}"
          puts "\n--------\n"
          min = 0.0
          max = 1.0
          precision = 2.to_i8
          in_bw = false
          prefixed = false
          reversed = false

          charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
          plot = charter.plot(net.error_distance_history, prefixed)

          puts "  plot: '#{plot}'"
          puts "  error_distance_history: '#{net.error_distance_history.map { |e| e.round(6) }}'"

          puts "\n--------\n"

          describe "JSON (de-)serialization works" do
            it "@calculated_error_total of the dumped net approximately matches @calculated_error_total of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation.from_json(json)

              assert_approximate_equality_of_nested_list net.calculated_error_total, net2.calculated_error_total, 0.000000001
            end

            it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation.from_json(json)

              assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
            end

            it "@weights of the dumped net approximately matches @weights of the loaded net" do
              json = net.to_json
              net2 = Ai4cr::NeuralNetwork::Backpropagation.from_json(json)

              assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
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
        end
      end
    end
  end
end
