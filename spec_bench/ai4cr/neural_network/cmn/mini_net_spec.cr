require "../../../spec_bench_helper"
require "../../../support/neural_network/data/*"

describe Ai4cr::NeuralNetwork::Cmn::MiniNet do
  describe "#train" do
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

      qty = MULTI_TYPE_TEST_QTY
      qty_x_percent = qty // QTY_X_PERCENT_DENOMINATOR

      [
        Ai4cr::NeuralNetwork::LS_PRELU,
        Ai4cr::NeuralNetwork::LS_RELU,
        Ai4cr::NeuralNetwork::LS_SIGMOID,
        Ai4cr::NeuralNetwork::LS_TANH,
      ].each do |learning_style|
        net = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 256, width: 3, history_size: 60, learning_style: learning_style)

        describe "and training #{qty} times each at a learning rate of #{net.learning_rate.round(6)} using learning_style: #{learning_style}" do
          puts_debug "\nTRAINING (learning_style: #{learning_style}):\n"
          qty.times do |i|
            print "." if i % 1000 == 0
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

          puts_debug "\n--------\n"
          min = 0.0
          max = 1.0
          precision = 2.to_i8
          in_bw = false
          prefixed = false
          reversed = false

          charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
          plot = charter.plot(net.error_stats.history, prefixed)

          puts_debug "#{net.class.name}:"
          puts_debug "  plot: '#{plot}'"
          puts_debug "  error_stats.history: '#{net.error_stats.history.map { |e| e.round(6) }}'"

          puts_debug "\n--------\n"

          # describe "JSON (de-)serialization works" do
          #   it "@error_stats.distance of the dumped net approximately matches @error_stats.distance of the loaded net" do
          #     json = net.to_json
          #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

          #     assert_approximate_equality_of_nested_list net.error_stats.distance, net2.error_stats.distance, 0.000000001
          #   end

          #   it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
          #     json = net.to_json
          #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

          #     assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
          #   end

          #   it "@weights of the dumped net approximately matches @weights of the loaded net" do
          #     json = net.to_json
          #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

          #     assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
          #   end
          # end

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
                next_guess = mini_net_relu_best_guess(net, tr_input)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = mini_net_relu_best_guess(net, sq_input)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = mini_net_relu_best_guess(net, cr_input)
                check_guess(next_guess, "CROSS")
              end
            end

            describe "noisy input data for" do
              it "TRIANGLE" do
                next_guess = mini_net_relu_best_guess(net, tr_with_noise)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = mini_net_relu_best_guess(net, sq_with_noise)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = mini_net_relu_best_guess(net, cr_with_noise)
                check_guess(next_guess, "CROSS")
              end
            end

            describe "base noisy input data for" do
              it "TRIANGLE" do
                next_guess = mini_net_relu_best_guess(net, tr_with_base_noise)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = mini_net_relu_best_guess(net, sq_with_base_noise)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = mini_net_relu_best_guess(net, cr_with_base_noise)
                check_guess(next_guess, "CROSS")
              end
            end
          end
        end
      end
    end
  end
end
