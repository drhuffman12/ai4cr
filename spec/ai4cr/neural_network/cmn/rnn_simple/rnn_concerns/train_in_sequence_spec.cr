# require "./../../../../../spec_helper"
# require "./../../../../../spectator_helper"
# require "./../../../../../support/hard_coded_weights.cr"

# Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainInSequence do
#   let(time_col_qty) { 4 }
#   let(io_offset) { time_col_qty }

#   let(input_size) { 21 }
#   let(output_size) { 21 }
#   let(hidden_layer_qty) { 1 }

#   let(deriv_scale) { 0.1 }
#   let(learning_rate) { 0.2 }
#   let(momentum) { 0.3 }

#   let(rnn_simple) {
#     Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
#       io_offset: io_offset,
#       time_col_qty: time_col_qty,

#       input_size: input_size,
#       output_size: output_size,
#       hidden_layer_qty: hidden_layer_qty,

#       deriv_scale: deriv_scale,
#       learning_rate: learning_rate,
#       momentum: momentum,
#     )
#   }

#   before_each do
#     rnn_simple.init_network

#     weights = HARD_CODED_WEIGHTS

#     rnn_simple.synaptic_layer_indexes.map do |li|
#       rnn_simple.time_col_indexes.map do |ti|
#         rnn_simple.mini_net_set[li][ti].weights = weights[li][ti]
#       end
#     end
#   end

#   let(steps) { 20 }
#   let(scale) { 100 }

#   let(sine_data) {
#     (0..2*steps).to_a.map do |i|
#       theta = (2 * Math::PI * (i / steps.to_f))
#       alt = Math.sin(theta)
#       # ((alt + 1) / (2))
#       alt
#     end
#   }

#   let(sine_data_state_values) { rnn_simple.float_to_state_values(sine_data) }
#   let(io_pairs) { rnn_simple.split_for_training(sine_data_state_values) }

#   describe "#train_in_sequence" do
#     let(expected_sequence_errors_first) {
#       [2459.6639586146703, 280.6566328972941, 5564.395245824211, 562.2117018314473, 21808.951824860105, 214.43983573080126, 453.6364391489499, 261.8921894168977, 1098.7981108036345, 157.73263361982535, 262.3141122390676, 400.4972930165538, 1498.7595200578157, 8.302281424033362, 260.1278249740025, 12571.156919462654, 226.77136189434688, 1329.9794133201044, 7.582634202177085, 16.477210828398, 365.6656807012697, 248.4370351025708, 406.62981563347165, 1110.690365518917, 227.82316303304623, 849.3507271285372, 451.9018715689745, 84.3758604104283, 25.189579254505457, 394.76566160514244, 57.07689879498523, 2162.509407451166, 819.0527493064855, 212.84969650129227]
#     }
#     let(expected_sequence_errors_first_sum) { expected_sequence_errors_first.sum }

#     let(expected_sequence_errors_second) {
#       [1158.5684962540001, 3705.86973222972, 4324.87049762353, 2706.5719707229787, 60982.13036531214, 234.16125470739863, 723.1676936494353, 1940.3495286837028, 154.81738374531506, 13.939703756640503, 3225.9937688733653, 123.11945917989082, 346.943388876915, 528.7785399945964, 1853.2007953783088, 2131.637476137365, 887.9449698501596, 77.39936827129011, 3240.5708805830504, 372.8936440631522, 1998.9991685058897, 3683.7918552056294, 431895.33068074286, 5287.807105962807, 1.95877465367901, 191.5622877380288, 5561.580502787875, 908.5798618808858, 17.201450332386315, 11.588621811485776, 23601.308200865366, 1864.6186710836519, 1721.2175583459357, 31.31629311921329]
#     }
#     let(expected_sequence_errors_second_sum) { expected_sequence_errors_second.sum }

#     let(expected_sequence_errors_third) {
#       [31.4973070165688, 76.63824891620229, 29.455531942124978, 240.31142160812905, 2161.81151037897, 284.13153344001995, 8009.83862192208, 163.6937427152591, 39.109377940851864, 10.987508559053945, 87.63816452568994, 601.0305470741387, 13484.728816575027, 362.3521001776191, 10.677902689396927, 3400.8915654223706, 711.648739065413, 82303.76520948927, 1333.4725261310132, 8277.896943724993, 689.4561619109297, 6.92246985735003, 694.495769419714, 1739.5559643321274, 57.59615421055133, 105.42917139575675, 1038.5049783356221, 43.79856588452074, 0.8540591149548298, 143.11132728674352, 5.530934291732107, 9806.52654074093, 593.3864779326468, 906.4806416018891]
#     }
#     let(expected_sequence_errors_third_sum) { expected_sequence_errors_third.sum }

#     let(expected_sequence_errors_nth) {
#       [4444.432561902472, 7899.510377908705, 1076.7572723164906, 2544.787922015254, 846.2667160909116, 1984.8793872654426, 916.279561128405, 172.7339453527526, 13.393718924719108, 13.89665413466466, 45.249330781046545, 1160.6704105813756, 6660.410011415686, 148795.1461140195, 221.05231791761238, 20277.4235320691, 310.12923953799185, 2164.382715467695, 337.9267381304983, 352.7297878686671, 454.5642949659361, 20538.333949578573, 5821.585814933556, 1370.8616060518418, 3706.851860310994, 35.31158218920687, 2396.435631773931, 730.8824713584401, 1873.2832959417474, 20729.540252214745, 4639.4387248425855, 33.41395314666324, 2412.6577935166497, 373.21458199138937]
#     }
#     let(expected_sequence_errors_nth_sum) { expected_sequence_errors_nth.sum }

#     context "after first session of sequenced training" do
#       pending "returns expected sequence of errors" do
#         sequence_errors = rnn_simple.train_in_sequence(io_pairs)

#         # puts
#         # puts "sequence_errors (first): #{sequence_errors}"
#         # puts "sequence_errors.sum (first): #{sequence_errors.sum}"
#         # puts

#         expect(sequence_errors).to eq(expected_sequence_errors_first)
#       end
#     end

#     context "after second session of sequenced training" do
#       pending "returns expected sequence of errors" do
#         rnn_simple.train_in_sequence(io_pairs)
#         sequence_errors = rnn_simple.train_in_sequence(io_pairs)

#         # puts
#         # puts "sequence_errors (second): #{sequence_errors}"
#         # puts "sequence_errors.sum (second): #{sequence_errors.sum}"
#         # puts

#         expect(sequence_errors).to eq(expected_sequence_errors_second)
#       end

#       context "sum of errors decreased compared to" do
#         pending "first" do
#           expect(expected_sequence_errors_second_sum).to be < expected_sequence_errors_first_sum
#         end
#       end
#     end

#     context "after third session of sequenced training" do
#       # NOTE: Not all RNN's are equal! Some do better than others.
#       # TODO: Find param and rnd seeds that make this succeed (have lowest sum of errors) after Nth training session (and adjust test data as applicable).
#       pending "returns expected sequence of errors" do
#         n = 2
#         n.times { rnn_simple.train_in_sequence(io_pairs) }
#         sequence_errors = rnn_simple.train_in_sequence(io_pairs)

#         # puts
#         # puts "sequence_errors (third): #{sequence_errors}"
#         # puts "sequence_errors.sum (third): #{sequence_errors.sum}"
#         # puts

#         expect(sequence_errors).to eq(expected_sequence_errors_third)
#       end

#       context "sum of errors decreased compared to" do
#         pending "first" do
#           expect(expected_sequence_errors_nth_sum).to be < expected_sequence_errors_first_sum
#         end

#         it "second" do
#           expect(expected_sequence_errors_nth_sum).to be < expected_sequence_errors_second_sum
#         end
#       end
#     end

#     context "after Nth session of sequenced training" do
#       # TODO: Not all RNN's are equal! Find param and rnd seeds that make this succeed (have lowest sum of errors) after Nth training session (and adjust test data as applicable).
#       pending "returns expected sequence of errors" do
#         n = 3
#         n.times { rnn_simple.train_in_sequence(io_pairs) }
#         sequence_errors = rnn_simple.train_in_sequence(io_pairs)

#         # puts
#         # puts "sequence_errors (Nth): #{sequence_errors}"
#         # puts "sequence_errors.sum (Nth): #{sequence_errors.sum}"
#         # puts

#         expect(sequence_errors).to eq(expected_sequence_errors_nth)
#       end

#       context "sum of errors decreased compared to" do
#         pending "first" do
#           expect(expected_sequence_errors_nth_sum).to be < expected_sequence_errors_first_sum
#         end

#         it "second" do
#           expect(expected_sequence_errors_nth_sum).to be < expected_sequence_errors_second_sum
#         end

#         pending "third" do
#           expect(expected_sequence_errors_nth_sum).to be < expected_sequence_errors_third_sum
#         end
#       end
#     end
#   end

#   describe "#shifted_inputs" do
#     let(io_pair_last) {
#       {
#         ins: [
#           # Last ins:
#           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         ],
#         outs: [
#           # Last outs
#           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         ],
#       }
#     }

#     let(expected_input_next) {
#       [
#         # Next ins:
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       ]
#     }

#     let(inputs_next) { rnn_simple.shifted_inputs(io_pair_last) }

#     it "returns expected values" do
#       expect(inputs_next).to eq(expected_input_next)
#     end
#   end

#   ####
#   describe "#train_and_guess_in_sequence" do
#     context "returns" do
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

#       # # Last ins:
#       # [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

#       # # Last outs
#       # [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#       # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

#       let(expected_next_guesses) {
#         [
#           [
#             [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.007333698390068671, 0.0, 1.0, 0.014446876037148561, 0.0, 1.0, 1.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1473927139433336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.21332089655613545, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24181703429192705, 0.0, 0.0, 0.7395108407671145, 0.06274850836626245, 0.0, 0.0, 0.44422913330964564, 0.06944673088811304, 0.0, 0.05419585119950241, 0.0],
#             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           ],
#           [
#             [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.007333698390068671, 0.0, 1.0, 0.014446876037148561, 0.0, 1.0, 1.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1473927139433336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.5154831740247104, 0.0, 0.0, 0.0, 0.43204922039048393, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           ],
#           [
#             [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.007333698390068671, 0.0, 1.0, 0.014446876037148561, 0.0, 1.0, 1.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.7999884279648168, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3632908553136538, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.9809306175089775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#           ],
#         ]
#       }

#       let(should_be_next_values) {
#         # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#         # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#         # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#         # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],

#         [
#           [
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#           ],
#           [
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#           ],
#           [
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#           ],
#         ]
#       }

#       context "after training" do
#         context "one training_round" do
#           pending "compared to expected_next_guesses" do
#             next_guesses = rnn_simple.train_and_guess_in_sequence(io_pairs)

#             expect(next_guesses).to eq(expected_next_guesses)
#           end
#         end

#         context "N training_rounds" do
#           # NOTE: Not all RNN's are equal! Some do better than others.
#           # TODO: Find param and rnd seeds that make this succeed (have lowest sum of errors) after Nth training session (and adjust test data as applicable).

#           pending "compared to should_be_next_values" do
#             training_rounds = 2 # 10000
#             next_guesses = rnn_simple.train_and_guess_in_sequence(io_pairs, training_rounds: training_rounds)

#             puts
#             puts "sine_data_state_values: #{sine_data_state_values.to_json}"
#             puts "next_guesses: #{next_guesses.to_json}"
#             puts

#             expect(next_guesses).to eq(should_be_next_values)
#           end
#         end
#       end
#     end
#   end
# end
