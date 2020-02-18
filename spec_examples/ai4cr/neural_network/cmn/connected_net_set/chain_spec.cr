require "json"
require "ascii_bar_charter"
require "./../../../../spec_helper"
require "../../../../support/neural_network/data/*"

def mini_net_exp_best_guess(net, raw_in)
  # result = net.eval(raw_in)
  # result.map { |v| v.round(6) }

  net.eval(raw_in)
  net.guesses_best
end

describe Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain do
  describe "#train" do
    describe "with a shape of [256,300,3]" do
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

        net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: 256, width: 300, error_distance_history_max: 60)
        net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(height: 300, width: 3, error_distance_history_max: 60)
        
        arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
        arr << net0
        arr << net1
        cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

        # net.learning_rate = rand
        qty = 500
        qty_10_percent = qty // 10

        describe "and training #{qty} times each at a learning rate of #{cns.net_set.last.learning_rate.round(6)}" do
          puts "\nTRAINING:\n"
          qty.times do |i|
            print "." if i % qty_10_percent == 0 # 1000 == 0
            errors = {} of Symbol => Float64
            [:tr, :sq, :cr].shuffle.each do |s|
              case s
              when :tr
                errors[:tr] = cns.train(tr_input, is_a_triangle)
                # net0.step_calculate_error_distance_history if i % qty_10_percent == 0
                cns.net_set.last.step_calculate_error_distance_history if i % qty_10_percent == 0
              when :sq
                errors[:sq] = cns.train(sq_input, is_a_square)
                # net0.step_calculate_error_distance_history if i % qty_10_percent == 0
                cns.net_set.last.step_calculate_error_distance_history if i % qty_10_percent == 0
              when :cr
                errors[:cr] = cns.train(cr_input, is_a_cross)
                # net0.step_calculate_error_distance_history if i % qty_10_percent == 0
                cns.net_set.last.step_calculate_error_distance_history if i % qty_10_percent == 0
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

          charter = AsciiBarCharter.new(min, max, precision, in_bw, reversed)
          plot = charter.plot(cns.net_set.last.error_distance_history, prefixed)

          puts "#{cns.class.name} with structure of #{cns.structure} with nets of types #{cns.net_set.map { |n| n.class.name }}:"
          puts "  plot: '#{plot}'"
          puts "  error_distance_history: '#{cns.net_set.last.error_distance_history.map { |e| e.round(6) }}'"

          puts "\n--------\n"

          # describe "JSON (de-)serialization works" do
          #   it "@calculated_error_total of the dumped net approximately matches @calculated_error_total of the loaded net" do
          #     json = net.to_json
          #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.from_json(json)

          #     assert_approximate_equality_of_nested_list net.calculated_error_total, net2.calculated_error_total, 0.000000001
          #   end

          #   it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
          #     json = net.to_json
          #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.from_json(json)

          #     assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
          #   end

          #   it "@weights of the dumped net approximately matches @weights of the loaded net" do
          #     json = net.to_json
          #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.from_json(json)

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
                next_guess = mini_net_exp_best_guess(cns, tr_input)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = mini_net_exp_best_guess(cns, sq_input)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = mini_net_exp_best_guess(cns, cr_input)
                check_guess(next_guess, "CROSS")
              end
            end

            describe "noisy input data for" do
              it "TRIANGLE" do
                next_guess = mini_net_exp_best_guess(cns, tr_with_noise)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = mini_net_exp_best_guess(cns, sq_with_noise)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = mini_net_exp_best_guess(cns, cr_with_noise)
                check_guess(next_guess, "CROSS")
              end
            end

            describe "base noisy input data for" do
              it "TRIANGLE" do
                next_guess = mini_net_exp_best_guess(cns, tr_with_base_noise)
                check_guess(next_guess, "TRIANGLE")
              end

              it "SQUARE" do
                next_guess = mini_net_exp_best_guess(cns, sq_with_base_noise)
                check_guess(next_guess, "SQUARE")
              end

              it "CROSS" do
                next_guess = mini_net_exp_best_guess(cns, cr_with_base_noise)
                check_guess(next_guess, "CROSS")
              end
            end
          end
        end
      end
    end
  end

  describe "when given a mix of Exp, Relu, and Tanh MiniNets all chained together (with associated IO sizes" do
    ne = Ai4cr::NeuralNetwork::Cmn::MiniNet::Exp.new(height: 3, width: 2)
    nr = Ai4cr::NeuralNetwork::Cmn::MiniNet::Relu.new(height: 2, width: 3)
    nt = Ai4cr::NeuralNetwork::Cmn::MiniNet::Tanh.new(height: 3, width: 4)

    arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
    arr << ne
    arr << nr
    arr << nt
    cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)

    initial_inputs = [rand, rand, rand]
    expected_inital_outputs = (arr.last.width.times.to_a.map { 0.0 })

    it "is valid" do
      puts "*"*8

      puts "cns: #{cns.pretty_inspect}"
      puts "*"*8

      puts "cns.validate!: #{cns.validate!}"
      puts "*"*8
      (cns.validate!).should be_true
    end

    it "updates last net's outputs when guessing" do
      cns.net_set.each { |net| net.init_network }

      (cns.guesses_best).should eq(expected_inital_outputs)

      cns.eval(initial_inputs)

      (cns.guesses_best).should_not eq(expected_inital_outputs)
    end
  end
end
