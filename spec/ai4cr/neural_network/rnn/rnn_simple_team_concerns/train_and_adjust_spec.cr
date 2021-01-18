# require "./../../../../spec_helper"
# require "./../../../../spectator_helper"

# Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::TrainAndAdjust do
#   let(rnn_simple_team) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeam.new }

#   let(input_set_given) {
#     [
#       [0.1, 0.2],
#       [0.3, 0.4],
#     ]
#   }

#   describe "#train" do
#     let(expected_outputs_guessed_before) { [[0.0], [0.0]] }
#     let(expected_outputs_guessed) {
#       [[0.14193], [0.20547]]
#     }
#     let(expected_outputs_guessed_team_before) {
#       rnn_simple_team.team_size.times.to_a.map { expected_outputs_guessed_before }
#     }

#     context "with hard-coded weights" do
#       let(expected_outputs_guessed) {
#         [[0.119], [0.09780000000000003]]
#       }
#       let(expected_outputs_guessed_team) {
#         rnn_simple_team.team_size.times.to_a.map { expected_outputs_guessed }
#       }

#       before_each do
#         weights = [
#           [
#             [
#               [-0.4, -0.3, -0.2],
#               [-0.1, 0.0, 0.1],
#               [0.2, 0.3, 0.4],
#             ],
#             [
#               [-0.4, -0.3, -0.2],
#               [-0.1, 0.0, 0.1],
#               [0.2, 0.3, 0.4],
#               [-0.4, -0.3, -0.2],
#               [-0.1, 0.0, 0.1],
#               [0.2, 0.3, 0.4],
#             ],
#           ],
#           [
#             [
#               [-0.2],
#               [0.1],
#               [0.3],
#             ],
#             [
#               [-0.4],
#               [-0.2],
#               [0.2],
#               [0.4],
#             ],
#           ],
#         ]
#         rnn_simple_team.team_members.each do |rnn_simple|
#           rnn_simple.synaptic_layer_indexes.map do |li|
#             rnn_simple.time_col_indexes.map do |ti|
#               rnn_simple.mini_net_set[li][ti].weights = weights[li][ti]
#             end
#           end
#         end
#       end

#       context "before" do
#         it "outputs_guessed start off all zero's" do
#           expect(rnn_simple_team.outputs_guessed).to eq(expected_outputs_guessed_team_before)
#         end
#       end

#       context "after" do
#         pending "calculates differing outputs" do
#           # it "calculates expected outputs" do
#           rnn_simple_team.train(input_set_given, expected_outputs_guessed)

#           puts
#           puts "rnn_simple_team.plot_error_distance_history:"
#           rnn_simple_team.plot_error_distance_history.each { |h| puts h }
#           puts

#           assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
#         end

#         let(expected_error_history) {
#           a = Array(Float64).new
#           [a, a, a, a, a, a, a, a, a, a]
#         }

#         it "does add to 'error_distance_history' (since just doing an 'train')" do
#           rnn_simple_team.train(input_set_given, expected_outputs_guessed)

#           puts
#           puts "rnn_simple_team.plot_error_distance_history:"
#           rnn_simple_team.plot_error_distance_history.each { |h| puts h }
#           puts
#           puts "rnn_simple_team.error_distance_history: #{rnn_simple_team.error_distance_history}"
#           puts
#           eh_combined = rnn_simple_team.plot_error_distance_history.zip(rnn_simple_team.error_distance_history)
#           puts "eh_combined: #{eh_combined}"
#           puts
#           eh_combined.each { |eh| puts eh[0]; puts eh[1]; puts }
#           puts

#           expect(rnn_simple_team.error_distance_history.size).to be >= 0
#           expect(rnn_simple_team.error_distance_history).not_to eq(expected_error_history)
#           expect(rnn_simple_team.error_distance_history.first).to be <= rnn_simple_team.error_distance_history.last

#           # assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
#         end
#       end
#     end

#     context "with random weights" do
#       context "before" do
#         it "outputs_guessed start off all zero's" do
#           expect(rnn_simple_team.outputs_guessed).to eq(expected_outputs_guessed_team_before)
#         end
#       end

#       context "after" do
#         it "calculates differing outputs per team member" do
#           rnn_simple_team.train(input_set_given, expected_outputs_guessed)

#           puts
#           puts "rnn_simple_team.outputs_guessed: #{rnn_simple_team.outputs_guessed.pretty_inspect}"
#           puts

#           # assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
#           rnn_simple_team.team_members.map_with_index do |rnn_simple_i, i|
#             rnn_simple_team.team_members.map_with_index do |rnn_simple_j, j|
#               return nil if i == j

#               # rnn_simple_i.outputs_guessed == rnn_simple_j.outputs_guessed
#               assert_approximate_inequality_of_nested_list(rnn_simple_i.outputs_guessed, rnn_simple_j.outputs_guessed)
#             end
#           end
#         end

#         let(expected_error_history) {
#           a = Array(Float64).new
#           [a, a, a, a, a, a, a, a, a, a]
#         }

#         it "does add to 'error_distance_history' (since just doing an 'train')" do
#           rnn_simple_team.train(input_set_given, expected_outputs_guessed)

#           puts
#           puts "rnn_simple_team.plot_error_distance_history:"
#           rnn_simple_team.plot_error_distance_history.each { |h| puts h }
#           puts
#           puts "rnn_simple_team.error_distance_history: #{rnn_simple_team.error_distance_history}"
#           puts

#           # expect(rnn_simple_team.error_distance_history).to eq(expected_error_history)
#           expect(rnn_simple_team.error_distance_history.size).to be >= 0
#           expect(rnn_simple_team.error_distance_history).not_to eq(expected_error_history)
#           expect(rnn_simple_team.error_distance_history.first).to be <= rnn_simple_team.error_distance_history.last

#           # assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
#         end
#       end
#     end
#   end
# end
