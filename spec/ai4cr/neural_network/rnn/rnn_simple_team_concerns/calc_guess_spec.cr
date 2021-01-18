require "./../../../../spec_helper"
require "./../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeamConcerns::CalcGuess do
  let(rnn_simple_team) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeam.new }

  let(input_set_given) {
    [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
  }

  describe "#eval" do
    let(expected_outputs_guessed_before) { [[0.0], [0.0]] }
    let(expected_outputs_guessed_team_before) {
      rnn_simple_team.team_size.times.to_a.map { expected_outputs_guessed_before }
    }

    context "with hard-coded weights" do
      let(expected_outputs_guessed) {
        [[0.119], [0.09780000000000003]]
      }
      let(expected_outputs_guessed_team) {
        rnn_simple_team.team_size.times.to_a.map { expected_outputs_guessed }
      }

      before_each do
        weights = [
          [
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
            ],
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
            ],
          ],
          [
            [
              [-0.2],
              [0.1],
              [0.3],
            ],
            [
              [-0.4],
              [-0.2],
              [0.2],
              [0.4],
            ],
          ],
        ]
        rnn_simple_team.team_members.each do |rnn_simple|
          rnn_simple.synaptic_layer_indexes.map do |li|
            rnn_simple.time_col_indexes.map do |ti|
              rnn_simple.mini_net_set[li][ti].weights = weights[li][ti]
            end
          end
        end
      end

      context "before" do
        it "outputs_guessed start off all zero's" do
          expect(rnn_simple_team.outputs_guessed).to eq(expected_outputs_guessed_team_before)
        end
      end

      context "after" do
        it "calculates expected outputs" do
          rnn_simple_team.eval(input_set_given)

          puts
          puts "rnn_simple_team.plot_error_distance_history: #{rnn_simple_team.plot_error_distance_history}"
          puts

          assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
        end

        let(expected_error_history) {
          a = Array(Float64).new
          [a, a, a, a, a, a, a, a, a, a]
        }

        it "does not add to 'error_distance_history' (since just doing an 'eval')" do
          rnn_simple_team.eval(input_set_given)

          puts
          puts "rnn_simple_team.plot_error_distance_history: #{rnn_simple_team.plot_error_distance_history}"
          puts
          puts "rnn_simple_team.error_distance_history: #{rnn_simple_team.error_distance_history}"
          puts

          expect(rnn_simple_team.error_distance_history).to eq(expected_error_history)

          # assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
        end
      end
    end

    context "with random weights" do
      context "before" do
        it "outputs_guessed start off all zero's" do
          expect(rnn_simple_team.outputs_guessed).to eq(expected_outputs_guessed_team_before)
        end
      end

      context "after" do
        it "calculates differing outputs per team member" do
          rnn_simple_team.eval(input_set_given)

          puts
          puts "rnn_simple_team.outputs_guessed: #{rnn_simple_team.outputs_guessed.pretty_inspect}"
          puts

          # assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
          rnn_simple_team.team_members.map_with_index do |rnn_simple_i, i|
            rnn_simple_team.team_members.map_with_index do |rnn_simple_j, j|
              return nil if i == j

              # rnn_simple_i.outputs_guessed == rnn_simple_j.outputs_guessed
              assert_approximate_inequality_of_nested_list(rnn_simple_i.outputs_guessed, rnn_simple_j.outputs_guessed)
            end
          end
        end

        let(expected_error_history) {
          a = Array(Float64).new
          [a, a, a, a, a, a, a, a, a, a]
        }

        it "does not add to 'error_distance_history' (since just doing an 'eval')" do
          rnn_simple_team.eval(input_set_given)

          puts
          puts "rnn_simple_team.plot_error_distance_history: #{rnn_simple_team.plot_error_distance_history}"
          puts
          puts "rnn_simple_team.error_distance_history: #{rnn_simple_team.error_distance_history}"
          puts

          expect(rnn_simple_team.error_distance_history).to eq(expected_error_history)

          # assert_approximate_equality_of_nested_list(expected_outputs_guessed_team, rnn_simple_team.outputs_guessed)
        end
      end
    end
  end

  # describe "#all_mini_net_outputs" do
  #   context "with hard-coded weights" do
  #     let(rnn_simple_team) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeam.new }

  #     let(input_set_given) {
  #       [
  #         [0.1, 0.2],
  #         [0.3, 0.4],
  #       ]
  #     }

  #     let(expected_all_mini_net_outputs_before) {
  #       [
  #         [
  #           [0.0, 0.0, 0.0],
  #           [0.0, 0.0, 0.0],
  #         ],
  #         [
  #           [0.0],
  #           [0.0],
  #         ],
  #       ]
  #     }

  #     let(expected_all_mini_net_outputs_after) {
  #       # TODO: manually verify calc's.
  #       #   For now, we'll assume accuracy and move onto
  #       #   verifying via some training sessions with
  #       #   some 'real data'
  #       [
  #         [
  #           [0.14, 0.27, 0.4],
  #           [0.0, 0.17099999999999999, 0.42200000000000004],
  #         ],
  #         [
  #           [0.119],
  #           [0.09780000000000003],
  #         ],
  #       ]
  #     }

  #     before_each do
  #       weights = [
  #         [
  #           [
  #             [-0.4, -0.3, -0.2],
  #             [-0.1, 0.0, 0.1],
  #             [0.2, 0.3, 0.4],
  #           ],
  #           [
  #             [-0.4, -0.3, -0.2],
  #             [-0.1, 0.0, 0.1],
  #             [0.2, 0.3, 0.4],
  #             [-0.4, -0.3, -0.2],
  #             [-0.1, 0.0, 0.1],
  #             [0.2, 0.3, 0.4],
  #           ],
  #         ],
  #         [
  #           [
  #             [-0.2],
  #             [0.1],
  #             [0.3],
  #           ],
  #           [
  #             [-0.4],
  #             [-0.2],
  #             [0.2],
  #             [0.4],
  #           ],
  #         ],
  #       ]
  #       rnn_simple_team.synaptic_layer_indexes.map do |li|
  #         rnn_simple_team.time_col_indexes.map do |ti|
  #           rnn_simple_team.mini_net_set[li][ti].weights = weights[li][ti]
  #         end
  #       end
  #     end

  #     context "before #eval" do
  #       it "returns all-zero outputs" do
  #         expect(rnn_simple_team.all_mini_net_outputs).to eq(expected_all_mini_net_outputs_before)
  #       end
  #     end

  #     context "after #eval" do
  #       it "returns expected non-zero outputs" do
  #         rnn_simple_team.eval(input_set_given)

  #         assert_approximate_equality_of_nested_list(expected_all_mini_net_outputs_after, rnn_simple_team.all_mini_net_outputs)
  #       end
  #     end
  #   end
  # end
end
