# TODO: refactor Chain based on code cleanup doen for RnnSimple; then, refactor this test file (w/ Spectator)

# require "../../../spec_bench_helper"
# require "../../../support/neural_network/data/*"

# describe Ai4cr::NeuralNetwork::Cmn::Chain do
#   # TODO: Review and then revise?/fix? Chain after Finishing Rnn!
#   describe "#train" do
#     hidden_size = 500
#     describe "with a shape of [256,#{hidden_size},#{hidden_size},3]" do
#       describe "using image data (input) and shape flags (output) for triangle, square, and cross" do
#         error_averages = [] of Float64
#         is_a_triangle = [1.0, 0.0, 0.0]
#         is_a_square = [0.0, 1.0, 0.0]
#         is_a_cross = [0.0, 0.0, 1.0]

#         tr_input = TRIANGLE.flatten.map { |input| input.to_f / 5.0 }
#         sq_input = SQUARE.flatten.map { |input| input.to_f / 5.0 }
#         cr_input = CROSS.flatten.map { |input| input.to_f / 5.0 }

#         tr_with_noise = TRIANGLE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
#         sq_with_noise = SQUARE_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }
#         cr_with_noise = CROSS_WITH_NOISE.flatten.map { |input| input.to_f / 5.0 }

#         tr_with_base_noise = TRIANGLE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
#         sq_with_base_noise = SQUARE_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }
#         cr_with_base_noise = CROSS_WITH_BASE_NOISE.flatten.map { |input| input.to_f / 5.0 }

#         # net.learning_rate = rand
#         qty = MULTI_TYPE_TEST_QTY
#         qty_x_percent = qty // QTY_X_PERCENT_DENOMINATOR

#         describe "using net of types of: Sigmoid" do
#           net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 256, width: hidden_size, history_size: 60, bias_disabled: false, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID)
#           net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: hidden_size, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID)
#           net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: 3, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID)

#           arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
#           arr << net0
#           arr << net1
#           arr << net2
#           cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

#           puts "\n--------\n"
#           puts "#{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}:"

#           describe "using #{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}" do
#             describe "and training #{qty} times each at a learning rate of #{cns.net_set.last.learning_rate.round(6)}" do
#               puts "\nTRAINING:\n"
#               timestamp_before = Time.utc
#               qty.times do |i|
#                 print "." if i % qty_x_percent == 0 # 1000 == 0
#                 errors = {} of Symbol => Float64
#                 [:tr, :sq, :cr].shuffle.each do |s|
#                   case s
#                   when :tr
#                     errors[:tr] = cns.train(tr_input, is_a_triangle)
#                   when :sq
#                     errors[:sq] = cns.train(sq_input, is_a_square)
#                   when :cr
#                     errors[:cr] = cns.train(cr_input, is_a_cross)
#                   end
#                 end
#                 error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
#               end
#               timestamp_after = Time.utc

#               puts "\n--------\n"
#               puts "duration: #{timestamp_after - timestamp_before}"
#               puts "\n--------\n"
#               min = 0.0
#               max = 1.0
#               precision = 2.to_i8
#               in_bw = false
#               prefixed = false
#               reversed = false

#               charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
#               plot = charter.plot(cns.net_set.last.error_stats.history, prefixed)

#               puts "#{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}:"
#               puts "  plot: '#{plot}'"
#               puts "  error_stats.history: '#{cns.net_set.last.error_stats.history.map { |e| e.round(6) }}'"

#               puts "\n--------\n"

#               # describe "JSON (de-)serialization works" do
#               #   it "@error_stats.distance of the dumped net approximately matches @error_stats.distance of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.error_stats.distance, net2.error_stats.distance, 0.000000001
#               #   end

#               #   it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
#               #   end

#               #   it "@weights of the dumped net approximately matches @weights of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
#               #   end
#               # end

#               describe "error_averages" do
#                 it "decrease (i.e.: first > last)" do
#                   (error_averages.first > error_averages.last).should eq(true)
#                 end

#                 it "should end up close to 0.1 +/- 0.1" do
#                   assert_approximate_equality(error_averages.last, 0.1, 0.1)
#                 end

#                 it "should end up close to 0.01 +/- 0.01" do
#                   assert_approximate_equality(error_averages.last, 0.01, 0.01)
#                 end

#                 it "should end up close to 0.001 +/- 0.001" do
#                   assert_approximate_equality(error_averages.last, 0.001, 0.001)
#                 end
#               end

#               describe "#eval correctly guesses shape flags (output) when given image data (input) of" do
#                 describe "original input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_input)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_input)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_input)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "base noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_base_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_base_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_base_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end
#               end
#             end
#           end
#         end

#         describe "using net of types of: (mixed: Relu, Prelu, Sigmoid)" do
#           net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 256, width: hidden_size, history_size: 60, bias_disabled: false, learning_style: Ai4cr::NeuralNetwork::LS_RELU)
#           net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: hidden_size, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_PRELU)
#           net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: 3, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID)

#           arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
#           arr << net0
#           arr << net1
#           arr << net2
#           cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

#           puts "\n--------\n"
#           puts "#{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}:"

#           describe "using #{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}" do
#             describe "and training #{qty} times each at a learning rate of #{cns.net_set.last.learning_rate.round(6)}" do
#               puts "\nTRAINING:\n"
#               timestamp_before = Time.utc
#               qty.times do |i|
#                 print "." if i % qty_x_percent == 0 # 1000 == 0
#                 errors = {} of Symbol => Float64
#                 [:tr, :sq, :cr].shuffle.each do |s|
#                   case s
#                   when :tr
#                     errors[:tr] = cns.train(tr_input, is_a_triangle)
#                   when :sq
#                     errors[:sq] = cns.train(sq_input, is_a_square)
#                   when :cr
#                     errors[:cr] = cns.train(cr_input, is_a_cross)
#                   end
#                 end
#                 error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
#               end
#               timestamp_after = Time.utc

#               puts "\n--------\n"
#               puts "duration: #{timestamp_after - timestamp_before}"
#               puts "\n--------\n"
#               min = 0.0
#               max = 1.0
#               precision = 2.to_i8
#               in_bw = false
#               prefixed = false
#               reversed = false

#               charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
#               plot = charter.plot(cns.net_set.last.error_stats.history, prefixed)

#               puts "  plot: '#{plot}'"
#               puts "  error_stats.history: '#{cns.net_set.last.error_stats.history.map { |e| e.round(6) }}'"

#               puts "\n--------\n"

#               # describe "JSON (de-)serialization works" do
#               #   it "@error_stats.distance of the dumped net approximately matches @error_stats.distance of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.error_stats.distance, net2.error_stats.distance, 0.000000001
#               #   end

#               #   it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
#               #   end

#               #   it "@weights of the dumped net approximately matches @weights of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
#               #   end
#               # end

#               describe "error_averages" do
#                 it "decrease (i.e.: first > last)" do
#                   (error_averages.first > error_averages.last).should eq(true)
#                 end

#                 it "should end up close to 0.1 +/- 0.1" do
#                   assert_approximate_equality(error_averages.last, 0.1, 0.1)
#                 end

#                 it "should end up close to 0.01 +/- 0.01" do
#                   assert_approximate_equality(error_averages.last, 0.01, 0.01)
#                 end

#                 it "should end up close to 0.001 +/- 0.001" do
#                   assert_approximate_equality(error_averages.last, 0.001, 0.001)
#                 end
#               end

#               describe "#eval correctly guesses shape flags (output) when given image data (input) of" do
#                 describe "original input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_input)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_input)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_input)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "base noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_base_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_base_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_base_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end
#               end
#             end
#           end
#         end

#         describe "using net of types of: (mixed: Relu, Relu, Sigmoid)" do
#           net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 256, width: hidden_size, history_size: 60, bias_disabled: false, learning_style: Ai4cr::NeuralNetwork::LS_RELU)
#           net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: hidden_size, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_RELU)
#           net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: 3, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID)

#           arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
#           arr << net0
#           arr << net1
#           arr << net2
#           cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

#           puts "\n--------\n"
#           puts "#{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}:"

#           describe "using #{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}" do
#             describe "and training #{qty} times each at a learning rate of #{cns.net_set.last.learning_rate.round(6)}" do
#               puts "\nTRAINING:\n"
#               timestamp_before = Time.utc
#               qty.times do |i|
#                 print "." if i % qty_x_percent == 0 # 1000 == 0
#                 errors = {} of Symbol => Float64
#                 [:tr, :sq, :cr].shuffle.each do |s|
#                   case s
#                   when :tr
#                     errors[:tr] = cns.train(tr_input, is_a_triangle)
#                   when :sq
#                     errors[:sq] = cns.train(sq_input, is_a_square)
#                   when :cr
#                     errors[:cr] = cns.train(cr_input, is_a_cross)
#                   end
#                 end
#                 error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
#               end
#               timestamp_after = Time.utc

#               puts "\n--------\n"
#               puts "duration: #{timestamp_after - timestamp_before}"
#               puts "\n--------\n"
#               min = 0.0
#               max = 1.0
#               precision = 2.to_i8
#               in_bw = false
#               prefixed = false
#               reversed = false

#               charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
#               plot = charter.plot(cns.net_set.last.error_stats.history, prefixed)

#               puts "  plot: '#{plot}'"
#               puts "  error_stats.history: '#{cns.net_set.last.error_stats.history.map { |e| e.round(6) }}'"

#               puts "\n--------\n"

#               # describe "JSON (de-)serialization works" do
#               #   it "@error_stats.distance of the dumped net approximately matches @error_stats.distance of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.error_stats.distance, net2.error_stats.distance, 0.000000001
#               #   end

#               #   it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
#               #   end

#               #   it "@weights of the dumped net approximately matches @weights of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
#               #   end
#               # end

#               describe "error_averages" do
#                 it "decrease (i.e.: first > last)" do
#                   (error_averages.first > error_averages.last).should eq(true)
#                 end

#                 it "should end up close to 0.1 +/- 0.1" do
#                   assert_approximate_equality(error_averages.last, 0.1, 0.1)
#                 end

#                 it "should end up close to 0.01 +/- 0.01" do
#                   assert_approximate_equality(error_averages.last, 0.01, 0.01)
#                 end

#                 it "should end up close to 0.001 +/- 0.001" do
#                   assert_approximate_equality(error_averages.last, 0.001, 0.001)
#                 end
#               end

#               describe "#eval correctly guesses shape flags (output) when given image data (input) of" do
#                 describe "original input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_input)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_input)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_input)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "base noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_base_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_base_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_base_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end
#               end
#             end
#           end
#         end

#         describe "using net of types of: (mixed: Relu, Relu, Relu)" do
#           net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 256, width: hidden_size, history_size: 60, bias_disabled: false, learning_style: Ai4cr::NeuralNetwork::LS_RELU)
#           net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: hidden_size, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_RELU)
#           net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: 3, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_RELU)

#           arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
#           arr << net0
#           arr << net1
#           arr << net2
#           cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

#           puts "\n--------\n"
#           puts "#{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}:"

#           describe "using #{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}" do
#             describe "and training #{qty} times each at a learning rate of #{cns.net_set.last.learning_rate.round(6)}" do
#               puts "\nTRAINING:\n"
#               timestamp_before = Time.utc
#               qty.times do |i|
#                 print "." if i % qty_x_percent == 0 # 1000 == 0
#                 errors = {} of Symbol => Float64
#                 [:tr, :sq, :cr].shuffle.each do |s|
#                   case s
#                   when :tr
#                     errors[:tr] = cns.train(tr_input, is_a_triangle)
#                   when :sq
#                     errors[:sq] = cns.train(sq_input, is_a_square)
#                   when :cr
#                     errors[:cr] = cns.train(cr_input, is_a_cross)
#                   end
#                 end
#                 error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
#               end
#               timestamp_after = Time.utc

#               puts "\n--------\n"
#               puts "duration: #{timestamp_after - timestamp_before}"
#               puts "\n--------\n"
#               min = 0.0
#               max = 1.0
#               precision = 2.to_i8
#               in_bw = false
#               prefixed = false
#               reversed = false

#               charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
#               plot = charter.plot(cns.net_set.last.error_stats.history, prefixed)

#               puts "  plot: '#{plot}'"
#               puts "  error_stats.history: '#{cns.net_set.last.error_stats.history.map { |e| e.round(6) }}'"

#               puts "\n--------\n"

#               # describe "JSON (de-)serialization works" do
#               #   it "@error_stats.distance of the dumped net approximately matches @error_stats.distance of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.error_stats.distance, net2.error_stats.distance, 0.000000001
#               #   end

#               #   it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
#               #   end

#               #   it "@weights of the dumped net approximately matches @weights of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
#               #   end
#               # end

#               describe "error_averages" do
#                 it "decrease (i.e.: first > last)" do
#                   (error_averages.first > error_averages.last).should eq(true)
#                 end

#                 it "should end up close to 0.1 +/- 0.1" do
#                   assert_approximate_equality(error_averages.last, 0.1, 0.1)
#                 end

#                 it "should end up close to 0.01 +/- 0.01" do
#                   assert_approximate_equality(error_averages.last, 0.01, 0.01)
#                 end

#                 it "should end up close to 0.001 +/- 0.001" do
#                   assert_approximate_equality(error_averages.last, 0.001, 0.001)
#                 end
#               end

#               describe "#eval correctly guesses shape flags (output) when given image data (input) of" do
#                 describe "original input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_input)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_input)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_input)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "base noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_base_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_base_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_base_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end
#               end
#             end
#           end
#         end

#         describe "using net of types of: (mixed: Sigmoid, Relu, Relu)" do
#           net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: 256, width: hidden_size, history_size: 60, bias_disabled: false, learning_style: Ai4cr::NeuralNetwork::LS_SIGMOID)
#           net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: hidden_size, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_RELU)
#           net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.new(height: hidden_size, width: 3, history_size: 60, bias_disabled: true, learning_style: Ai4cr::NeuralNetwork::LS_RELU)

#           arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
#           arr << net0
#           arr << net1
#           arr << net2
#           cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)

#           puts "\n--------\n"
#           puts "#{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}:"

#           describe "using #{cns.class.name} with structure of #{cns.structure} with nets of learning styles #{cns.net_set.map { |n| n.learning_style }}" do
#             describe "and training #{qty} times each at a learning rate of #{cns.net_set.last.learning_rate.round(6)}" do
#               puts "\nTRAINING:\n"
#               timestamp_before = Time.utc
#               qty.times do |i|
#                 print "." if i % qty_x_percent == 0 # 1000 == 0
#                 errors = {} of Symbol => Float64
#                 [:tr, :sq, :cr].shuffle.each do |s|
#                   case s
#                   when :tr
#                     errors[:tr] = cns.train(tr_input, is_a_triangle)
#                   when :sq
#                     errors[:sq] = cns.train(sq_input, is_a_square)
#                   when :cr
#                     errors[:cr] = cns.train(cr_input, is_a_cross)
#                   end
#                 end
#                 error_averages << (errors[:tr].to_f + errors[:sq].to_f + errors[:cr].to_f) / 3.0
#               end
#               timestamp_after = Time.utc

#               puts "\n--------\n"
#               puts "duration: #{timestamp_after - timestamp_before}"
#               puts "\n--------\n"
#               min = 0.0
#               max = 1.0
#               precision = 2.to_i8
#               in_bw = false
#               prefixed = false
#               reversed = false

#               charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
#               plot = charter.plot(cns.net_set.last.error_stats.history, prefixed)

#               puts "  plot: '#{plot}'"
#               puts "  error_stats.history: '#{cns.net_set.last.error_stats.history.map { |e| e.round(6) }}'"

#               puts "\n--------\n"

#               # describe "JSON (de-)serialization works" do
#               #   it "@error_stats.distance of the dumped net approximately matches @error_stats.distance of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.error_stats.distance, net2.error_stats.distance, 0.000000001
#               #   end

#               #   it "@activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.activation_nodes, net2.activation_nodes, 0.000000001
#               #   end

#               #   it "@weights of the dumped net approximately matches @weights of the loaded net" do
#               #     json = net.to_json
#               #     net2 = Ai4cr::NeuralNetwork::Cmn::MiniNet.from_json(json)

#               #     assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
#               #   end
#               # end

#               describe "error_averages" do
#                 it "decrease (i.e.: first > last)" do
#                   (error_averages.first > error_averages.last).should eq(true)
#                 end

#                 it "should end up close to 0.1 +/- 0.1" do
#                   assert_approximate_equality(error_averages.last, 0.1, 0.1)
#                 end

#                 it "should end up close to 0.01 +/- 0.01" do
#                   assert_approximate_equality(error_averages.last, 0.01, 0.01)
#                 end

#                 it "should end up close to 0.001 +/- 0.001" do
#                   assert_approximate_equality(error_averages.last, 0.001, 0.001)
#                 end
#               end

#               describe "#eval correctly guesses shape flags (output) when given image data (input) of" do
#                 describe "original input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_input)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_input)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_input)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end

#                 describe "base noisy input data for" do
#                   it "TRIANGLE" do
#                     next_guess = mini_net_exp_best_guess(cns, tr_with_base_noise)
#                     check_guess(next_guess, "TRIANGLE")
#                   end

#                   it "SQUARE" do
#                     next_guess = mini_net_exp_best_guess(cns, sq_with_base_noise)
#                     check_guess(next_guess, "SQUARE")
#                   end

#                   it "CROSS" do
#                     next_guess = mini_net_exp_best_guess(cns, cr_with_base_noise)
#                     check_guess(next_guess, "CROSS")
#                   end
#                 end
#               end
#             end
#           end
#         end
#       end
#     end
#   end
# end
