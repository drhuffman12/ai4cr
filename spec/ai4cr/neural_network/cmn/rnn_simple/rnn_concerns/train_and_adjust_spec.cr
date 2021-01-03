require "./../../../../../spec_helper"
require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainAndAdjust do
  let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

  let(input_set_given) {
    [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
  }

  describe "#train" do
    let(expected_outputs_guessed_before) { [[0.0], [0.0]] }
    let(expected_outputs_trained) { [[0.4], [0.6]] }

    context "with hard-coded weights" do
      let(expected_outputs_guessed) {
        [[0.119], [0.0656]]
      }
      let(hard_coded_weights) {
        [
          [
            [
              [-0.42, -0.32, -0.22],
              [-0.12, 0.025, 0.12],
              [0.22, 0.32, 0.42],
            ],
            [
              [-0.41, -0.31, -0.21],
              [-0.11, 0.06, 0.11],
              [0.21, 0.31, 0.41],
              [-0.4, -0.3, -0.2],
              [-0.1, 0.05, 0.1],
              [0.2, 0.3, 0.4],
            ],
          ],
          [
            [
              [-0.23],
              [0.13],
              [0.33],
            ],
            [
              [-0.44],
              [-0.24],
              [0.24],
              [0.44],
            ],
          ],
        ]
      }

      before_each do
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            rnn_simple.mini_net_set[li][ti].weights = hard_coded_weights[li][ti].clone
          end
        end
      end

      context "before" do
        it "outputs_guessed start off all zero's" do
          expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_before)
        end
      end

      context "during" do
        pending "calls 'step_load_inputs'" do
          expect(rnn_simple).to receive(:step_load_inputs)

          rnn_simple.train(input_set_given, expected_outputs_trained)
        end

        pending "calls 'step_calc_forward'" do
          expect(rnn_simple).to receive(:step_calc_forward)

          rnn_simple.train(input_set_given, expected_outputs_trained)
        end

        pending "calls 'step_load_outputs'" do
          expect(rnn_simple).to receive(:step_load_outputs)

          rnn_simple.train(input_set_given, expected_outputs_trained)
        end

        pending "calls 'step_calculate_error'" do
          expect(rnn_simple).to receive(:step_calculate_error)

          rnn_simple.train(input_set_given, expected_outputs_trained)
        end

        pending "calls 'step_backpropagate'" do
          expect(rnn_simple).to receive(:step_backpropagate)

          rnn_simple.train(input_set_given, expected_outputs_trained)
        end
      end

      context "after" do
        let(expected_error_total) { 0.18227218 }

        it "returns expected error_total" do
          error_total = rnn_simple.train(input_set_given, expected_outputs_trained)

          expect(error_total).to eq(expected_error_total)
        end

        it "calculates expected outputs" do
          rnn_simple.train(input_set_given, expected_outputs_trained)

          assert_approximate_equality_of_nested_list(expected_outputs_guessed, rnn_simple.outputs_guessed)
        end

        it "adjusts weights" do
          rnn_simple.train(input_set_given, expected_outputs_trained)

          puts "----"
          puts "hard_coded_weights: #{hard_coded_weights}"
          puts "VS"
          puts "rnn_simple.all_mini_net_weights: #{rnn_simple.all_mini_net_weights}"
          puts "----"

          assert_approximate_inequality_of_nested_list(hard_coded_weights, rnn_simple.all_mini_net_weights, 0.000_000_1)
        end
      end
    end
  end

  # describe "#all_mini_net_outputs" do
  #   context "with hard-coded weights" do
  #     let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

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
  #           [0.04000000000000001, 0.21, 0.38],
  #         ],
  #         [
  #           [0.119],
  #           [0.0656],
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
  #       rnn_simple.synaptic_layer_indexes.map do |li|
  #         rnn_simple.time_col_indexes.map do |ti|
  #           rnn_simple.mini_net_set[li][ti].weights = weights[li][ti]
  #         end
  #       end
  #     end

  #     context "before #eval" do
  #       it "returns all-zero outputs" do
  #         expect(rnn_simple.all_mini_net_outputs).to eq(expected_all_mini_net_outputs_before)
  #       end
  #     end

  #     context "after #eval" do
  #       it "returns expected non-zero outputs" do
  #         rnn_simple.eval(input_set_given)

  #         assert_approximate_equality_of_nested_list(expected_all_mini_net_outputs_after, rnn_simple.all_mini_net_outputs)
  #       end
  #     end
  #   end
  # end
end
